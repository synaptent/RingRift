"""
Distributed training infrastructure for local Mac cluster and cloud deployment.

This module provides:
- Worker discovery via Bonjour/mDNS
- HTTP client for distributed task execution
- Coordinator utilities for CMA-ES population evaluation
- Queue abstractions for cloud deployment (Redis, SQS)
- In-memory game collection for distributed recording
"""

from .discovery import (
    WorkerDiscovery,
    WorkerInfo,
    discover_workers,
    wait_for_workers,
    parse_manual_workers,
    filter_healthy_workers,
)
from .client import (
    WorkerClient,
    DistributedEvaluator,
    QueueDistributedEvaluator,
    EvaluationStats,
)
from .queue import (
    TaskQueue,
    EvalTask,
    EvalResult,
    GameReplayData,
    RedisQueue,
    SQSQueue,
    get_task_queue,
)
from .game_collector import (
    InMemoryGameCollector,
    CollectedGame,
    deserialize_game_data,
    write_games_to_db,
)

__all__ = [
    # Local cluster (HTTP-based)
    "WorkerDiscovery",
    "WorkerInfo",
    "discover_workers",
    "wait_for_workers",
    "parse_manual_workers",
    "filter_healthy_workers",
    "WorkerClient",
    "DistributedEvaluator",
    "QueueDistributedEvaluator",
    "EvaluationStats",
    # Cloud deployment (queue-based)
    "TaskQueue",
    "EvalTask",
    "EvalResult",
    "GameReplayData",
    "RedisQueue",
    "SQSQueue",
    "get_task_queue",
    # Game recording
    "InMemoryGameCollector",
    "CollectedGame",
    "deserialize_game_data",
    "write_games_to_db",
]
