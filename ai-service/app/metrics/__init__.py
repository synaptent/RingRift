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
from typing import Optional

logger = logging.getLogger(__name__)

# Import orchestrator metrics
# Re-export app-level metrics from the standalone metrics module
# These are used by app/main.py for API request metrics
import sys
import importlib.util

# Load app/metrics.py directly (it exists alongside this package)
_metrics_file = __file__.replace("__init__.py", "").rstrip("/").rstrip("\\") + ".py"
_spec = importlib.util.spec_from_file_location("app.metrics_base", _metrics_file.replace("/metrics/", "/metrics"))
if _spec and _spec.loader:
    # The actual file is at app/metrics.py (parent dir + metrics.py)
    import os
    _parent = os.path.dirname(os.path.dirname(__file__))
    _metrics_py = os.path.join(_parent, "metrics.py")
    if os.path.exists(_metrics_py):
        _spec = importlib.util.spec_from_file_location("_metrics_base", _metrics_py)
        if _spec and _spec.loader:
            _metrics_base = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_metrics_base)
            # Export the app-level metrics
            AI_MOVE_REQUESTS = _metrics_base.AI_MOVE_REQUESTS
            AI_MOVE_LATENCY = _metrics_base.AI_MOVE_LATENCY
            AI_INSTANCE_CACHE_SIZE = _metrics_base.AI_INSTANCE_CACHE_SIZE
            AI_INSTANCE_CACHE_LOOKUPS = _metrics_base.AI_INSTANCE_CACHE_LOOKUPS
            PYTHON_INVARIANT_VIOLATIONS = _metrics_base.PYTHON_INVARIANT_VIOLATIONS
            observe_ai_move_start = _metrics_base.observe_ai_move_start

from app.metrics.orchestrator import (
    # Selfplay metrics
    SELFPLAY_GAMES_TOTAL,
    SELFPLAY_GAMES_PER_SECOND,
    SELFPLAY_BATCH_DURATION,
    SELFPLAY_ERRORS_TOTAL,
    SELFPLAY_QUEUE_SIZE,
    # Training metrics
    TRAINING_RUNS_TOTAL,
    TRAINING_RUN_DURATION,
    TRAINING_LOSS,
    TRAINING_ACCURACY,
    TRAINING_SAMPLES_PROCESSED,
    TRAINING_EPOCHS_TOTAL,
    # Evaluation metrics
    EVALUATION_GAMES_TOTAL,
    EVALUATION_ELO_DELTA,
    EVALUATION_WIN_RATE,
    EVALUATION_DURATION,
    # Promotion metrics
    MODEL_PROMOTIONS_TOTAL,
    MODEL_PROMOTION_ELO_GAIN,
    MODEL_PROMOTION_REJECTIONS,
    CURRENT_MODEL_ELO,
    # Pipeline metrics
    PIPELINE_STAGE_DURATION,
    PIPELINE_ITERATIONS_TOTAL,
    PIPELINE_ERRORS_TOTAL,
    PIPELINE_STATE,
    # Sync metrics
    DATA_SYNC_DURATION,
    DATA_SYNC_GAMES,
    DATA_SYNC_ERRORS,
    MODEL_SYNC_DURATION,
    MODEL_SYNC_TOTAL,
    # Helper functions
    record_selfplay_batch,
    record_training_run,
    record_evaluation,
    record_model_promotion,
    record_promotion_rejection,
    record_pipeline_stage,
    record_data_sync,
    record_model_sync,
    time_pipeline_stage,
    set_pipeline_state,
    # State constants
    PIPELINE_IDLE,
    PIPELINE_SELFPLAY,
    PIPELINE_TRAINING,
    PIPELINE_EVALUATION,
    PIPELINE_PROMOTION,
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
    # App-level metrics (from metrics.py)
    "AI_MOVE_REQUESTS",
    "AI_MOVE_LATENCY",
    "AI_INSTANCE_CACHE_SIZE",
    "AI_INSTANCE_CACHE_LOOKUPS",
    "PYTHON_INVARIANT_VIOLATIONS",
    "observe_ai_move_start",
    # Selfplay metrics
    "SELFPLAY_GAMES_TOTAL",
    "SELFPLAY_GAMES_PER_SECOND",
    "SELFPLAY_BATCH_DURATION",
    "SELFPLAY_ERRORS_TOTAL",
    "SELFPLAY_QUEUE_SIZE",
    # Training metrics
    "TRAINING_RUNS_TOTAL",
    "TRAINING_RUN_DURATION",
    "TRAINING_LOSS",
    "TRAINING_ACCURACY",
    "TRAINING_SAMPLES_PROCESSED",
    "TRAINING_EPOCHS_TOTAL",
    # Evaluation metrics
    "EVALUATION_GAMES_TOTAL",
    "EVALUATION_ELO_DELTA",
    "EVALUATION_WIN_RATE",
    "EVALUATION_DURATION",
    # Promotion metrics
    "MODEL_PROMOTIONS_TOTAL",
    "MODEL_PROMOTION_ELO_GAIN",
    "MODEL_PROMOTION_REJECTIONS",
    "CURRENT_MODEL_ELO",
    # Pipeline metrics
    "PIPELINE_STAGE_DURATION",
    "PIPELINE_ITERATIONS_TOTAL",
    "PIPELINE_ERRORS_TOTAL",
    "PIPELINE_STATE",
    # Sync metrics
    "DATA_SYNC_DURATION",
    "DATA_SYNC_GAMES",
    "DATA_SYNC_ERRORS",
    "MODEL_SYNC_DURATION",
    "MODEL_SYNC_TOTAL",
    # Helper functions
    "record_selfplay_batch",
    "record_training_run",
    "record_evaluation",
    "record_model_promotion",
    "record_promotion_rejection",
    "record_pipeline_stage",
    "record_data_sync",
    "record_model_sync",
    "time_pipeline_stage",
    "set_pipeline_state",
    # State constants
    "PIPELINE_IDLE",
    "PIPELINE_SELFPLAY",
    "PIPELINE_TRAINING",
    "PIPELINE_EVALUATION",
    "PIPELINE_PROMOTION",
    # Server
    "start_metrics_server",
    "is_metrics_server_running",
    # Training logger
    "create_training_logger",
]
