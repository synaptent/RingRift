"""
Training Scripts Library

Shared utilities for cluster operations, data processing, and monitoring.

Modules:
    alerts: Alert infrastructure for monitoring scripts
    cluster: Cluster node management and SSH operations
    config: Training configuration management
    database: Database connection and transaction utilities
    data_quality: Game quality scoring and filtering
    logging_config: Structured logging setup
    paths: Standard project paths and utilities
    retry: Retry decorators with exponential backoff
    validation: Data and model validation utilities

Usage:
    from scripts.lib.alerts import Alert, AlertManager, AlertSeverity, AlertType
    from scripts.lib.cluster import ClusterManager, ClusterNode
    from scripts.lib.config import get_config, TrainingConfig
    from scripts.lib.database import safe_transaction, get_game_db_path
    from scripts.lib.data_quality import GameQualityScorer, QualityFilter
    from scripts.lib.logging_config import setup_script_logging, get_logger
    from scripts.lib.paths import AI_SERVICE_ROOT, DATA_DIR, MODELS_DIR
    from scripts.lib.retry import retry, retry_on_exception, RetryConfig
    from scripts.lib.validation import validate_npz_file, DataValidator
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
    get_game_db_path,
    get_elo_db_path,
    count_games,
    table_exists,
    get_db_size_mb,
    vacuum_database,
    check_integrity,
    DATA_DIR,
    GAMES_DIR,
    UNIFIED_ELO_DB,
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

__all__ = [
    # logging_config
    "setup_logging",
    "setup_script_logging",
    "get_logger",
    "get_metrics_logger",
    "MetricsLogger",
    "JsonFormatter",
    "ColoredFormatter",
    # config
    "TrainingConfig",
    "ModelConfig",
    "BoardConfig",
    "ConfigManager",
    "get_config",
    "get_board_config",
    "get_config_manager",
    # validation
    "ValidationResult",
    "validate_npz_file",
    "validate_jsonl_file",
    "validate_model_file",
    "validate_training_config",
    "validate_cluster_health",
    "DataValidator",
    # alerts
    "Alert",
    "AlertSeverity",
    "AlertType",
    "AlertThresholds",
    "AlertManager",
    "create_alert",
    "check_disk_alert",
    "check_memory_alert",
    "check_cpu_alert",
    # cluster
    "ClusterManager",
    "ClusterNode",
    "ClusterAutomation",
    "VastNodeManager",
    "NodeStatus",
    "NodeHealth",
    "GPUInfo",
    "CommandResult",
    "CommandError",
    "get_cluster",
    "get_automation",
    # database
    "safe_transaction",
    "read_only_connection",
    "get_game_db_path",
    "get_elo_db_path",
    "count_games",
    "table_exists",
    "get_db_size_mb",
    "vacuum_database",
    "check_integrity",
    "DATA_DIR",
    "GAMES_DIR",
    "UNIFIED_ELO_DB",
    # retry
    "RetryConfig",
    "RetryAttempt",
    "retry",
    "retry_on_exception",
    "retry_async",
    "with_timeout",
    # paths
    "AI_SERVICE_ROOT",
    "DATA_DIR",
    "GAMES_DIR",
    "SELFPLAY_DIR",
    "TRAINING_DIR",
    "MODELS_DIR",
    "NNUE_MODELS_DIR",
    "LOGS_DIR",
    "CONFIG_DIR",
    "SCRIPTS_DIR",
    "UNIFIED_ELO_DB",
    "get_game_db_path",
    "get_training_data_path",
    "get_model_path",
    "get_log_path",
    "ensure_dir",
    "ensure_parent_dir",
]
