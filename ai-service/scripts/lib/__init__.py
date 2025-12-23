"""
Training Scripts Library

Shared utilities for cluster operations, data processing, and monitoring.

Modules:
    alerts: Alert infrastructure for monitoring scripts
    cli: CLI argument patterns and helpers
    cluster: Cluster node management and SSH operations
    config: Training configuration management
    database: Database connection and transaction utilities
    data_quality: Game quality scoring and filtering
    datetime_utils: Timestamp generation, parsing, and file age operations
    file_formats: JSONL/JSON handling with gzip detection
    hosts: Unified cluster host configuration (distributed_hosts.yaml)
    metrics: Statistics collection, timing, rates, and progress tracking
    state_manager: Persistent state loading/saving for daemon scripts
    health: System and service health monitoring
    logging_config: Structured logging setup
    paths: Standard project paths and utilities
    process: Process management, signals, and singleton locks
    retry: Retry decorators with exponential backoff
    transfer: File transfer utilities (SCP, rsync, checksums)
    validation: Data and model validation utilities

Usage:
    from scripts.lib.alerts import Alert, AlertManager, AlertSeverity, AlertType
    from scripts.lib.cluster import ClusterManager, ClusterNode
    from scripts.lib.config import get_config, TrainingConfig
    from scripts.lib.database import safe_transaction, get_game_db_path
    from scripts.lib.data_quality import GameQualityScorer, QualityFilter
    from scripts.lib.logging_config import setup_script_logging, get_logger
    from scripts.lib.paths import AI_SERVICE_ROOT, DATA_DIR, MODELS_DIR
    from scripts.lib.process import SingletonLock, SignalHandler, run_command
    from scripts.lib.retry import retry, retry_on_exception, RetryConfig
    from scripts.lib.transfer import TransferConfig, scp_push, rsync_pull
    from scripts.lib.health import SystemHealth, check_system_health, check_http_health
    from scripts.lib.cli import add_common_args, add_board_args, get_config_key
    from scripts.lib.file_formats import load_json, save_json, read_jsonl_lines
    from scripts.lib.validation import validate_npz_file, DataValidator
    from scripts.lib.datetime_utils import format_elapsed_time, timestamp_id, ElapsedTimer
    from scripts.lib.metrics import TimingStats, RateCalculator, ProgressTracker
    from scripts.lib.hosts import get_hosts, get_host, HostConfig, get_active_hosts
"""

from scripts.lib.logging_config import (
    setup_logging,
    setup_script_logging,
    get_logger,
    get_metrics_logger,
    MetricsLogger,
    JsonFormatter,
    ColoredFormatter,
)

from scripts.lib.config import (
    TrainingConfig,
    ModelConfig,
    BoardConfig,
    ConfigManager,
    get_config,
    get_board_config,
    get_config_manager,
)

from scripts.lib.validation import (
    ValidationResult,
    validate_npz_file,
    validate_jsonl_file,
    validate_model_file,
    validate_training_config,
    validate_cluster_health,
    DataValidator,
)

from scripts.lib.alerts import (
    Alert,
    AlertSeverity,
    AlertType,
    AlertThresholds,
    AlertManager,
    create_alert,
    check_disk_alert,
    check_memory_alert,
    check_cpu_alert,
)

from scripts.lib.data_quality import (
    VictoryType,
    VICTORY_TYPE_VALUE,
    GameLengthConfig,
    QualityScores,
    GameQuality,
    QualityWeights,
    GameQualityScorer,
    QualityFilter,
    QualityStats,
    compute_quality_stats,
)

from scripts.lib.cluster import (
    ClusterManager,
    ClusterNode,
    ClusterAutomation,
    VastNodeManager,
    NodeStatus,
    NodeHealth,
    GPUInfo,
    CommandResult,
    CommandError,
    get_cluster,
    get_automation,
)

from scripts.lib.database import (
    safe_transaction,
    read_only_connection,
    get_elo_db_path,
    count_games,
    table_exists,
    get_db_size_mb,
    vacuum_database,
    check_integrity,
)

from scripts.lib.retry import (
    RetryConfig,
    RetryAttempt,
    retry,
    retry_on_exception,
    retry_async,
    with_timeout,
)

from scripts.lib.paths import (
    AI_SERVICE_ROOT,
    DATA_DIR,
    GAMES_DIR,
    SELFPLAY_DIR,
    TRAINING_DIR,
    MODELS_DIR,
    NNUE_MODELS_DIR,
    LOGS_DIR,
    CONFIG_DIR,
    SCRIPTS_DIR,
    UNIFIED_ELO_DB,
    get_game_db_path,
    get_training_data_path,
    get_model_path,
    get_log_path,
    ensure_dir,
    ensure_parent_dir,
)

from scripts.lib.process import (
    SingletonLock,
    SignalHandler,
    ProcessInfo,
    CommandOutput,
    is_process_running,
    find_processes_by_pattern,
    count_processes_by_pattern,
    kill_process,
    kill_processes_by_pattern,
    run_command,
    daemon_context,
    wait_for_process_exit,
    get_process_start_time,
)

from scripts.lib.transfer import (
    TransferConfig,
    TransferResult,
    compute_checksum,
    copy_local,
    compress_file,
    decompress_file,
    scp_push,
    scp_pull,
    rsync_push,
    rsync_pull,
    verify_transfer,
)

from scripts.lib.health import (
    DiskHealth,
    MemoryHealth,
    CPUHealth,
    GPUInfo as HealthGPUInfo,
    SystemHealth,
    ServiceHealth,
    check_disk_space,
    check_memory,
    check_cpu,
    check_gpus,
    check_system_health,
    check_http_health,
    check_port_open,
    check_process_health,
    wait_for_healthy,
)

from scripts.lib.cli import (
    add_verbose_arg,
    add_dry_run_arg,
    add_config_arg,
    add_node_arg,
    add_board_args,
    add_output_arg,
    add_limit_arg,
    add_parallel_arg,
    add_timeout_arg,
    add_common_args,
    setup_cli_logging,
    get_config_key,
    parse_config_key,
    validate_path_arg,
    confirm_action,
    parse_board_type,
    BOARD_TYPES,
    BOARD_TYPE_MAP,
    VALID_PLAYER_COUNTS,
)

from scripts.lib.file_formats import (
    is_gzip_file,
    open_jsonl_file,
    read_jsonl_lines,
    count_jsonl_lines,
    write_jsonl_lines,
    load_json,
    load_json_strict,
    load_json_if_exists,
    save_json,
    save_json_compact,
    update_json,
    get_file_size_mb,
    get_uncompressed_size_estimate,
)

from scripts.lib.datetime_utils import (
    get_file_age,
    get_file_age_hours,
    get_file_age_days,
    is_file_older_than,
    find_files_older_than,
    iter_files_by_age,
    format_elapsed_time,
    format_elapsed_time_short,
    ElapsedTimer,
    timestamp_id,
    timestamp_id_ms,
    timestamp_for_log,
    timestamp_iso,
    timestamp_iso_utc,
    parse_timestamp,
    parse_timestamp_safe,
    timestamp_age,
)

from scripts.lib.metrics import (
    TimingStats,
    RateCalculator,
    Counter,
    WinLossCounter,
    ProgressTracker,
    RunningStats,
    MetricsCollection,
)

from scripts.lib.state_manager import (
    StateManager,
    StatePersistence,
    load_json_state,
    save_json_state,
)

from scripts.lib.hosts import (
    HostConfig,
    HostsManager,
    EloSyncConfig,
    get_hosts,
    get_host,
    get_host_names,
    get_elo_sync_config,
    get_hosts_by_group,
    get_hosts_manager,
    load_distributed_hosts,
    get_training_hosts,
    get_selfplay_hosts,
    get_active_hosts,
    get_p2p_voters,
)

from scripts.lib.unified_cluster_config import (
    NodeConfig,
    UnifiedClusterConfig,
    get_cluster_config,
    reload_cluster_config,
)

__all__ = [
    # paths
    "AI_SERVICE_ROOT",
    "BOARD_TYPES",
    "BOARD_TYPE_MAP",
    "CONFIG_DIR",
    "DATA_DIR",
    "DATA_DIR",
    "GAMES_DIR",
    "GAMES_DIR",
    "LOGS_DIR",
    "MODELS_DIR",
    "NNUE_MODELS_DIR",
    "SCRIPTS_DIR",
    "SELFPLAY_DIR",
    "TRAINING_DIR",
    "UNIFIED_ELO_DB",
    "UNIFIED_ELO_DB",
    "VALID_PLAYER_COUNTS",
    "VICTORY_TYPE_VALUE",
    # alerts
    "Alert",
    "AlertManager",
    "AlertSeverity",
    "AlertThresholds",
    "AlertType",
    "BoardConfig",
    "CPUHealth",
    "ClusterAutomation",
    # cluster
    "ClusterManager",
    "ClusterNode",
    "ColoredFormatter",
    "CommandError",
    "CommandOutput",
    "CommandResult",
    "ConfigManager",
    "Counter",
    "DataValidator",
    # health
    "DiskHealth",
    "ElapsedTimer",
    "EloSyncConfig",
    "GPUInfo",
    "GameLengthConfig",
    "GameQuality",
    "GameQualityScorer",
    "HealthGPUInfo",
    # hosts
    "HostConfig",
    "HostsManager",
    "JsonFormatter",
    "MemoryHealth",
    "MetricsCollection",
    "MetricsLogger",
    "ModelConfig",
    "NodeHealth",
    "NodeStatus",
    "ProcessInfo",
    "ProgressTracker",
    "QualityFilter",
    "QualityScores",
    "QualityStats",
    "QualityWeights",
    "RateCalculator",
    "RetryAttempt",
    # retry
    "RetryConfig",
    "RunningStats",
    "ServiceHealth",
    "SignalHandler",
    # process
    "SingletonLock",
    # state_manager
    "StateManager",
    "StatePersistence",
    "SystemHealth",
    # metrics
    "TimingStats",
    # config
    "TrainingConfig",
    # transfer
    "TransferConfig",
    "TransferResult",
    # validation
    "ValidationResult",
    "VastNodeManager",
    # data_quality
    "VictoryType",
    "WinLossCounter",
    "add_board_args",
    "add_common_args",
    "add_config_arg",
    "add_dry_run_arg",
    "add_limit_arg",
    "add_node_arg",
    "add_output_arg",
    "add_parallel_arg",
    "add_timeout_arg",
    # cli
    "add_verbose_arg",
    "check_cpu",
    "check_cpu_alert",
    "check_disk_alert",
    "check_disk_space",
    "check_gpus",
    "check_http_health",
    "check_integrity",
    "check_memory",
    "check_memory_alert",
    "check_port_open",
    "check_process_health",
    "check_system_health",
    "compress_file",
    "compute_checksum",
    "compute_quality_stats",
    "confirm_action",
    "copy_local",
    "count_games",
    "count_jsonl_lines",
    "count_processes_by_pattern",
    "create_alert",
    "daemon_context",
    "decompress_file",
    "ensure_dir",
    "ensure_parent_dir",
    "find_files_older_than",
    "find_processes_by_pattern",
    "format_elapsed_time",
    "format_elapsed_time_short",
    "get_active_hosts",
    "get_automation",
    "get_board_config",
    "get_cluster",
    "get_config",
    "get_config_key",
    "get_config_manager",
    "get_db_size_mb",
    "get_elo_db_path",
    "get_elo_sync_config",
    # datetime_utils
    "get_file_age",
    "get_file_age_days",
    "get_file_age_hours",
    "get_file_size_mb",
    "get_game_db_path",
    "get_game_db_path",
    "get_host",
    "get_host_names",
    "get_hosts",
    "get_hosts_by_group",
    "get_hosts_manager",
    "get_log_path",
    "get_logger",
    "get_metrics_logger",
    "get_model_path",
    "get_p2p_voters",
    "get_process_start_time",
    "get_selfplay_hosts",
    "get_training_data_path",
    "get_training_hosts",
    "get_uncompressed_size_estimate",
    "is_file_older_than",
    # file_formats
    "is_gzip_file",
    "is_process_running",
    "iter_files_by_age",
    "kill_process",
    "kill_processes_by_pattern",
    "load_distributed_hosts",
    "load_json",
    "load_json_if_exists",
    "load_json_state",
    "load_json_strict",
    "open_jsonl_file",
    "parse_board_type",
    "parse_config_key",
    "parse_timestamp",
    "parse_timestamp_safe",
    "read_jsonl_lines",
    "read_only_connection",
    "retry",
    "retry_async",
    "retry_on_exception",
    "rsync_pull",
    "rsync_push",
    "run_command",
    # database
    "safe_transaction",
    "save_json",
    "save_json_compact",
    "save_json_state",
    "scp_pull",
    "scp_push",
    "setup_cli_logging",
    # logging_config
    "setup_logging",
    "setup_script_logging",
    "table_exists",
    "timestamp_age",
    "timestamp_for_log",
    "timestamp_id",
    "timestamp_id_ms",
    "timestamp_iso",
    "timestamp_iso_utc",
    "update_json",
    "vacuum_database",
    "validate_cluster_health",
    "validate_jsonl_file",
    "validate_model_file",
    "validate_npz_file",
    "validate_path_arg",
    "validate_training_config",
    "verify_transfer",
    "wait_for_healthy",
    "wait_for_process_exit",
    "with_timeout",
    "write_jsonl_lines",
    # unified_cluster_config
    "NodeConfig",
    "UnifiedClusterConfig",
    "get_cluster_config",
    "reload_cluster_config",
]
