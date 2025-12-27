"""Unified metrics system for RingRift AI service.

This package consolidates all metrics collection for the AI service,
including:
- Prometheus metrics for monitoring and alerting
- Training metrics for experiment tracking
- Orchestrator metrics for pipeline observability

Usage:
    from app.metrics import (
        # Orchestrator metrics
        record_selfplay_batch,
        record_training_run,
        record_model_promotion,
        # Metrics server
        start_metrics_server,
    )

    # Record selfplay progress
    record_selfplay_batch(
        board_type="square8",
        num_players=2,
        games=100,
        duration_seconds=60.5,
    )

    # Start metrics server for Prometheus scraping
    start_metrics_server(port=9090)

Note: For the original app-level metrics (AI_MOVE_REQUESTS, etc.),
import directly from app.metrics_base (the renamed app/metrics.py).
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

# Dec 26, 2025: Eliminated dynamic importlib loading
# Now directly importing from app.metrics_base (renamed from app/metrics.py)
from app.metrics_base import (
    # Prometheus metric objects
    AI_ERRORS,
    AI_FALLBACKS,
    AI_INSTANCE_CACHE_LOOKUPS,
    AI_INSTANCE_CACHE_SIZE,
    AI_MOVE_LATENCY,
    AI_MOVE_REQUESTS,
    PYTHON_INVARIANT_VIOLATIONS,
    # Helper functions
    observe_ai_move_start,
    record_ai_error,
    record_ai_fallback,
    # Promotion metrics
    record_promotion_decision,
    record_promotion_execution,
    # Elo reconciliation metrics
    record_elo_drift,
    record_elo_sync,
)

# Import coordinator metrics
from app.metrics.coordinator import (
    # Bandwidth metrics
    BANDWIDTH_ALLOCATIONS_ACTIVE,
    BANDWIDTH_BYTES_TOTAL,
    BANDWIDTH_TRANSFERS_TOTAL,
    COORDINATOR_ERRORS,
    COORDINATOR_OPERATIONS,
    # Status metrics
    COORDINATOR_STATUS,
    COORDINATOR_UPTIME,
    JOBS_TRACKED,
    NODES_TRACKED,
    # Recovery metrics
    RECOVERY_ATTEMPTS,
    RECOVERY_ESCALATIONS,
    SYNC_CLUSTER_HEALTH,
    SYNC_GAMES_UNSYNCED,
    SYNC_HOSTS_CRITICAL,
    SYNC_HOSTS_HEALTHY,
    SYNC_HOSTS_STALE,
    # Sync metrics
    SYNC_HOSTS_TOTAL,
    collect_all_coordinator_metrics,
    collect_all_coordinator_metrics_sync,
    record_coordinator_error,
    record_coordinator_operation,
    update_bandwidth_stats,
    # Update functions
    update_coordinator_status,
    update_coordinator_uptime,
    update_recovery_stats,
    update_sync_stats,
)
from app.metrics.orchestrator import (
    CURRENT_MODEL_ELO,
    DATA_SERVER_STATUS,
    # Sync metrics
    DATA_SYNC_DURATION,
    DATA_SYNC_ERRORS,
    DATA_SYNC_GAMES,
    EVALUATION_DURATION,
    EVALUATION_ELO_DELTA,
    # Evaluation metrics
    EVALUATION_GAMES_TOTAL,
    EVALUATION_WIN_RATE,
    MODEL_PROMOTION_ELO_GAIN,
    MODEL_PROMOTION_REJECTIONS,
    # Promotion metrics
    MODEL_PROMOTIONS_TOTAL,
    MODEL_SYNC_DURATION,
    MODEL_SYNC_TOTAL,
    PIPELINE_ERRORS_TOTAL,
    PIPELINE_EVALUATION,
    # State constants
    PIPELINE_IDLE,
    PIPELINE_ITERATIONS_TOTAL,
    PIPELINE_PROMOTION,
    PIPELINE_SELFPLAY,
    # Pipeline metrics
    PIPELINE_STAGE_DURATION,
    PIPELINE_STATE,
    PIPELINE_TRAINING,
    QUALITY_BRIDGE_STATUS,
    QUALITY_SYNC_STATS,
    SELFPLAY_BATCH_DURATION,
    SELFPLAY_ERRORS_TOTAL,
    SELFPLAY_GAMES_PER_SECOND,
    # Selfplay metrics
    SELFPLAY_GAMES_TOTAL,
    SELFPLAY_QUEUE_SIZE,
    SYNC_COORDINATOR_BYTES,
    SYNC_COORDINATOR_DURATION,
    SYNC_COORDINATOR_ERRORS,
    SYNC_COORDINATOR_FILES,
    # Sync coordinator metrics
    SYNC_COORDINATOR_OPS,
    SYNC_NFS_SKIP,
    SYNC_SOURCES_DISCOVERED,
    TRAINING_ACCURACY,
    TRAINING_DATA_DECISIVE_RATIO,
    TRAINING_DATA_ELO,
    TRAINING_DATA_GAMES_TOTAL,
    TRAINING_DATA_HIGH_QUALITY_COUNT,
    TRAINING_DATA_QUALITY_HISTOGRAM,
    # Training data quality metrics
    TRAINING_DATA_QUALITY_SCORE,
    TRAINING_EPOCHS_TOTAL,
    TRAINING_LOSS,
    TRAINING_RUN_DURATION,
    # Training metrics
    TRAINING_RUNS_TOTAL,
    TRAINING_SAMPLES_PROCESSED,
    collect_quality_metrics_from_bridge,
    collect_quality_metrics_from_manifest,
    get_pipeline_iterations,
    get_selfplay_queue_size,
    record_data_sync,
    record_evaluation,
    record_high_quality_sync,
    record_model_promotion,
    record_model_sync,
    record_nfs_skip,
    record_pipeline_iteration,
    record_pipeline_stage,
    record_promotion_rejection,
    # Helper functions
    record_selfplay_batch,
    record_sync_coordinator_op,
    # Quality metrics helper functions
    record_training_data_quality,
    record_training_run,
    set_pipeline_state,
    time_pipeline_stage,
    update_data_server_status,
    update_quality_bridge_status,
    # Queue and iteration tracking (December 2025)
    update_selfplay_queue_size,
    update_sync_sources_count,
)

# Metrics server management
_server_started = False
_server_lock = threading.RLock()


def start_metrics_server(port: int = 9090) -> bool:
    """Start the Prometheus metrics HTTP server.

    This should be called once at application startup to expose
    metrics for Prometheus scraping.

    Args:
        port: HTTP port for the metrics server

    Returns:
        True if server started, False if already running
    """
    global _server_started

    with _server_lock:
        if _server_started:
            logger.debug(f"Metrics server already running on port {port}")
            return False

        try:
            from prometheus_client import start_http_server
            start_http_server(port)
            _server_started = True
            logger.info(f"Prometheus metrics server started on port {port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            return False


def is_metrics_server_running() -> bool:
    """Check if the metrics server is running."""
    return _server_started


# Convenience function to get training logger
def create_training_logger(*args, **kwargs):
    """Create a training metrics logger.

    See app.training.metrics_logger.create_training_logger for full documentation.
    """
    from app.training.metrics_logger import create_training_logger as _create
    return _create(*args, **kwargs)


__all__ = [
    "AI_ERRORS",
    "AI_FALLBACKS",
    "AI_INSTANCE_CACHE_LOOKUPS",
    "AI_INSTANCE_CACHE_SIZE",
    "AI_MOVE_LATENCY",
    # App-level metrics (from metrics.py)
    "AI_MOVE_REQUESTS",
    "BANDWIDTH_ALLOCATIONS_ACTIVE",
    "BANDWIDTH_BYTES_TOTAL",
    "BANDWIDTH_TRANSFERS_TOTAL",
    "COORDINATOR_ERRORS",
    "COORDINATOR_OPERATIONS",
    # Coordinator metrics
    "COORDINATOR_STATUS",
    "COORDINATOR_UPTIME",
    "CURRENT_MODEL_ELO",
    "DATA_SERVER_STATUS",
    # Sync metrics
    "DATA_SYNC_DURATION",
    "DATA_SYNC_ERRORS",
    "DATA_SYNC_GAMES",
    "EVALUATION_DURATION",
    "EVALUATION_ELO_DELTA",
    # Evaluation metrics
    "EVALUATION_GAMES_TOTAL",
    "EVALUATION_WIN_RATE",
    "JOBS_TRACKED",
    # Promotion metrics
    "MODEL_PROMOTIONS_TOTAL",
    "MODEL_PROMOTION_ELO_GAIN",
    "MODEL_PROMOTION_REJECTIONS",
    "MODEL_SYNC_DURATION",
    "MODEL_SYNC_TOTAL",
    "NODES_TRACKED",
    # Parity metrics (December 2025)
    "PARITY_HEALTHCHECK_CASES_TOTAL",
    "PARITY_HEALTHCHECK_PASS_RATE",
    "PARITY_MISMATCHES_TOTAL",
    "PIPELINE_ERRORS_TOTAL",
    "PIPELINE_EVALUATION",
    # State constants
    "PIPELINE_IDLE",
    "PIPELINE_ITERATIONS_TOTAL",
    "PIPELINE_PROMOTION",
    "PIPELINE_SELFPLAY",
    # Pipeline metrics
    "PIPELINE_STAGE_DURATION",
    "PIPELINE_STATE",
    "PIPELINE_TRAINING",
    "PYTHON_INVARIANT_VIOLATIONS",
    "QUALITY_BRIDGE_STATUS",
    "QUALITY_SYNC_STATS",
    "RECOVERY_ATTEMPTS",
    "RECOVERY_ESCALATIONS",
    "SELFPLAY_BATCH_DURATION",
    "SELFPLAY_ERRORS_TOTAL",
    "SELFPLAY_GAMES_PER_SECOND",
    # Selfplay metrics
    "SELFPLAY_GAMES_TOTAL",
    "SELFPLAY_QUEUE_SIZE",
    "SYNC_CLUSTER_HEALTH",
    "SYNC_COORDINATOR_BYTES",
    "SYNC_COORDINATOR_DURATION",
    "SYNC_COORDINATOR_ERRORS",
    "SYNC_COORDINATOR_FILES",
    # Sync coordinator metrics
    "SYNC_COORDINATOR_OPS",
    "SYNC_GAMES_UNSYNCED",
    "SYNC_HOSTS_CRITICAL",
    "SYNC_HOSTS_HEALTHY",
    "SYNC_HOSTS_STALE",
    "SYNC_HOSTS_TOTAL",
    "SYNC_NFS_SKIP",
    "SYNC_SOURCES_DISCOVERED",
    "TRAINING_ACCURACY",
    "TRAINING_DATA_DECISIVE_RATIO",
    "TRAINING_DATA_ELO",
    "TRAINING_DATA_GAMES_TOTAL",
    "TRAINING_DATA_HIGH_QUALITY_COUNT",
    "TRAINING_DATA_QUALITY_HISTOGRAM",
    # Training data quality metrics
    "TRAINING_DATA_QUALITY_SCORE",
    "TRAINING_EPOCHS_TOTAL",
    "TRAINING_LOSS",
    # Training metrics
    "TRAINING_RUNS_TOTAL",
    "TRAINING_RUN_DURATION",
    "TRAINING_SAMPLES_PROCESSED",
    # Catalog (December 2025)
    "MetricCatalog",
    "MetricCategory",
    "MetricInfo",
    "MetricType",
    "collect_all_coordinator_metrics",
    "collect_all_coordinator_metrics_sync",
    "collect_quality_metrics_from_bridge",
    "collect_quality_metrics_from_manifest",
    # Training logger
    "create_training_logger",
    "emit_parity_summary_metrics",
    "get_metric",
    "get_metric_catalog",
    "get_pipeline_iterations",
    "get_selfplay_queue_size",
    "is_metric_registered",
    "is_metrics_server_running",
    "list_registered_metrics",
    "observe_ai_move_start",
    "record_ai_error",
    "record_ai_fallback",
    "record_coordinator_error",
    "record_coordinator_operation",
    "record_data_sync",
    "record_elo_drift",
    "record_elo_sync",
    "record_evaluation",
    "record_high_quality_sync",
    "record_model_promotion",
    "record_model_sync",
    "record_nfs_skip",
    "record_parity_case",
    "record_parity_mismatch",
    "record_pipeline_iteration",
    "record_pipeline_stage",
    # Promotion and Elo reconciliation metrics
    "record_promotion_decision",
    "record_promotion_execution",
    "record_promotion_rejection",
    # Helper functions
    "record_selfplay_batch",
    "record_sync_coordinator_op",
    # Quality metrics helper functions
    "record_training_data_quality",
    "record_training_run",
    "register_metric",
    "safe_counter",
    "safe_gauge",
    "safe_histogram",
    # Registry (December 2025: Consolidated _safe_metric pattern)
    "safe_metric",
    "safe_summary",
    "set_pipeline_state",
    # Server
    "start_metrics_server",
    "time_pipeline_stage",
    "update_bandwidth_stats",
    "update_coordinator_status",
    "update_coordinator_uptime",
    "update_data_server_status",
    "update_parity_pass_rate",
    "update_quality_bridge_status",
    "update_recovery_stats",
    # Queue and iteration tracking (December 2025)
    "update_selfplay_queue_size",
    "update_sync_sources_count",
    "update_sync_stats",
]

# Import catalog (December 2025)
from app.metrics.catalog import (
    MetricCatalog,
    MetricCategory,
    MetricInfo,
    MetricType,
    get_metric_catalog,
    register_metric,
)

# Import parity metrics (December 2025)
from app.metrics.parity import (
    PARITY_HEALTHCHECK_CASES_TOTAL,
    PARITY_HEALTHCHECK_PASS_RATE,
    PARITY_MISMATCHES_TOTAL,
    emit_parity_summary_metrics,
    record_parity_case,
    record_parity_mismatch,
    update_parity_pass_rate,
)

# Import registry (December 2025: Consolidated _safe_metric pattern)
from app.metrics.registry import (
    get_metric,
    is_metric_registered,
    list_registered_metrics,
    safe_counter,
    safe_gauge,
    safe_histogram,
    safe_metric,
    safe_summary,
)
