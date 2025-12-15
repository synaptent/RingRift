"""
Distributed training infrastructure for local Mac cluster and cloud deployment.

This module provides:
- Worker discovery via Bonjour/mDNS
- HTTP client for distributed task execution
- Coordinator utilities for CMA-ES population evaluation
- Queue abstractions for cloud deployment (Redis, SQS)
- In-memory game collection for distributed recording
- Host configuration and memory detection for cluster operations
- Memory profiling and peak tracking utilities
"""

from .hosts import (
    HostConfig,
    HostMemoryInfo,
    load_remote_hosts,
    get_local_memory_gb,
    get_remote_memory_gb,
    detect_host_memory,
    detect_all_host_memory,
    get_eligible_hosts_for_board,
    get_high_memory_hosts,
    clear_memory_cache,
    SSHExecutor,
    get_ssh_executor,
    BOARD_MEMORY_REQUIREMENTS,
)
from .memory import (
    MemorySample,
    MemoryProfile,
    MemoryTracker,
    RemoteMemoryMonitor,
    get_current_rss_mb,
    get_peak_rss_mb,
    get_process_rss_mb,
    profile_function,
    format_memory_profile,
    write_memory_report,
)
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
from .cluster_coordinator import (
    ClusterCoordinator,
    TaskRole,
    ProcessLimits,
    TaskInfo,
    check_and_abort_if_role_held,
)
from .db_utils import (
    atomic_write,
    safe_transaction,
    exclusive_db_lock,
    atomic_json_update,
    TransactionManager,
    save_state_atomically,
    load_state_safely,
)
from .health_checks import (
    HealthChecker,
    HealthSummary,
    ComponentHealth,
    get_health_summary,
    format_health_report,
)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitStatus,
    CircuitOpenError,
    get_host_breaker,
    get_training_breaker,
    format_circuit_status,
)

__all__ = [
    # Host configuration and memory detection
    "HostConfig",
    "HostMemoryInfo",
    "load_remote_hosts",
    "get_local_memory_gb",
    "get_remote_memory_gb",
    "detect_host_memory",
    "detect_all_host_memory",
    "get_eligible_hosts_for_board",
    "get_high_memory_hosts",
    "clear_memory_cache",
    "SSHExecutor",
    "get_ssh_executor",
    "BOARD_MEMORY_REQUIREMENTS",
    # Memory profiling
    "MemorySample",
    "MemoryProfile",
    "MemoryTracker",
    "RemoteMemoryMonitor",
    "get_current_rss_mb",
    "get_peak_rss_mb",
    "get_process_rss_mb",
    "profile_function",
    "format_memory_profile",
    "write_memory_report",
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
    # Cluster coordination
    "ClusterCoordinator",
    "TaskRole",
    "ProcessLimits",
    "TaskInfo",
    "check_and_abort_if_role_held",
    # Database utilities
    "atomic_write",
    "safe_transaction",
    "exclusive_db_lock",
    "atomic_json_update",
    "TransactionManager",
    "save_state_atomically",
    "load_state_safely",
    # Health checks
    "HealthChecker",
    "HealthSummary",
    "ComponentHealth",
    "get_health_summary",
    "format_health_report",
]
