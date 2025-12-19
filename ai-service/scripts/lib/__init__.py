"""
Training Scripts Library

Shared utilities for cluster operations, data processing, and monitoring.

Modules:
    alerts: Alert infrastructure for monitoring scripts
    cluster: Cluster node management and SSH operations
    config: Training configuration management
    data_quality: Game quality scoring and filtering
    logging_config: Structured logging setup
    validation: Data and model validation utilities

Usage:
    from scripts.lib.alerts import Alert, AlertManager, AlertSeverity, AlertType
    from scripts.lib.cluster import ClusterManager, ClusterNode
    from scripts.lib.config import get_config, TrainingConfig
    from scripts.lib.data_quality import GameQualityScorer, QualityFilter
    from scripts.lib.logging_config import setup_script_logging, get_logger
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
    get_cluster,
    get_automation,
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
    "get_cluster",
    "get_automation",
]
