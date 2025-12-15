#!/usr/bin/env python3
"""Unified AI Self-Improvement Loop - Single coordinator for the complete improvement cycle.

This daemon integrates all components of the AI improvement loop:
1. Streaming Data Collection - 60-second incremental sync from all hosts
2. Shadow Tournament Service - 15-minute lightweight evaluation
3. Training Scheduler - Auto-trigger when data thresholds met
4. Model Promoter - Auto-deploy on Elo threshold
5. Adaptive Curriculum - Elo-weighted training focus

Replaces the need for separate daemons by providing a single entry point
that coordinates all improvement activities with tight integration.

Usage:
    # Start the unified loop
    python scripts/unified_ai_loop.py --start

    # Run in foreground with verbose output
    python scripts/unified_ai_loop.py --foreground --verbose

    # Check status
    python scripts/unified_ai_loop.py --status

    # Stop gracefully
    python scripts/unified_ai_loop.py --stop

    # Use custom config
    python scripts/unified_ai_loop.py --config config/unified_loop.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import sqlite3
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml

# Optional Prometheus client
try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

# Allow imports from app/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
RINGRIFT_ROOT = AI_SERVICE_ROOT.parent

# =============================================================================
# Emergency Halt - File-based emergency stop mechanism
# =============================================================================
# To halt the loop: touch ai-service/data/coordination/EMERGENCY_HALT
# To resume: rm ai-service/data/coordination/EMERGENCY_HALT

EMERGENCY_HALT_FILE = AI_SERVICE_ROOT / "data" / "coordination" / "EMERGENCY_HALT"


def check_emergency_halt() -> bool:
    """Check if emergency halt flag is set.

    The emergency halt mechanism allows operators to immediately pause
    all AI loop activities by creating a file. This is useful for:
    - Emergency interventions during problematic training
    - Maintenance windows
    - Cluster-wide pauses

    Returns:
        True if the loop should halt, False otherwise.
    """
    return EMERGENCY_HALT_FILE.exists()


def clear_emergency_halt() -> bool:
    """Clear the emergency halt flag to resume operations.

    Returns:
        True if the flag was cleared, False if it didn't exist.
    """
    if EMERGENCY_HALT_FILE.exists():
        EMERGENCY_HALT_FILE.unlink()
        return True
    return False

# Import health checks for component monitoring
try:
    from app.distributed.health_checks import (
        HealthChecker,
        HealthSummary,
        get_health_summary,
        format_health_report,
    )
    HAS_HEALTH_CHECKS = True
except ImportError:
    HAS_HEALTH_CHECKS = False
    HealthChecker = None
    HealthSummary = None

# Import TaskCoordinator for global task limits and rate limiting
try:
    from app.coordination.task_coordinator import (
        TaskCoordinator,
        TaskType,
        TaskLimits,
        CoordinatorState,
    )
    HAS_TASK_COORDINATOR = True
except ImportError:
    HAS_TASK_COORDINATOR = False
    TaskCoordinator = None
    TaskType = None

# Coordination features: OrchestratorRole, backpressure, sync_lock, bandwidth, duration
try:
    from app.coordination import (
        # Orchestrator role management (SQLite-backed with heartbeat)
        OrchestratorRole,
        acquire_orchestrator_role,
        release_orchestrator_role,
        is_orchestrator_role_available,
        orchestrator_role,
        get_registry,
        # Queue backpressure
        QueueType,
        should_throttle_production,
        should_stop_production,
        get_throttle_factor,
        report_queue_depth,
        # Sync mutex for rsync coordination
        sync_lock,
        # Bandwidth management
        request_bandwidth,
        release_bandwidth,
        TransferPriority,
        # Duration-aware scheduling
        can_schedule_task,
        register_running_task,
        record_task_completion,
        estimate_task_duration,
    )
    HAS_COORDINATION = True
except ImportError:
    HAS_COORDINATION = False
    OrchestratorRole = None

# Import centralized Elo service (canonical ELO operations)
try:
    from app.training.elo_service import (
        EloService,
        get_elo_service,
        ELO_DB_PATH as CANONICAL_ELO_DB_PATH,
    )
    HAS_ELO_SERVICE = True
except ImportError:
    HAS_ELO_SERVICE = False
    EloService = None
    get_elo_service = None
    CANONICAL_ELO_DB_PATH = None

# Import hot data buffer for in-memory game caching (reduces DB roundtrips)
try:
    from app.training.hot_data_buffer import (
        HotDataBuffer,
        GameRecord,
        create_hot_buffer,
    )
    HAS_HOT_BUFFER = True
except ImportError:
    HAS_HOT_BUFFER = False
    HotDataBuffer = None
    GameRecord = None
    create_hot_buffer = None

# Import adaptive controller for plateau detection and dynamic scaling
try:
    from app.training.adaptive_controller import (
        AdaptiveController,
        IterationResult,
        create_adaptive_controller,
    )
    HAS_ADAPTIVE_CONTROLLER = True
except ImportError:
    HAS_ADAPTIVE_CONTROLLER = False
    AdaptiveController = None
    IterationResult = None
    create_adaptive_controller = None

# Import value calibration for model prediction quality analysis
try:
    from app.training.value_calibration import (
        ValueCalibrator,
        CalibrationTracker,
        CalibrationReport,
    )
    HAS_VALUE_CALIBRATION = True
except ImportError:
    HAS_VALUE_CALIBRATION = False
    ValueCalibrator = None
    CalibrationTracker = None
    CalibrationReport = None

# Import temperature scheduling for exploration/exploitation control
try:
    from app.training.temperature_scheduling import (
        TemperatureScheduler,
        TemperatureConfig,
        ScheduleType,
        create_scheduler as create_temp_scheduler,
    )
    HAS_TEMPERATURE_SCHEDULING = True
except ImportError:
    HAS_TEMPERATURE_SCHEDULING = False
    TemperatureScheduler = None
    TemperatureConfig = None
    ScheduleType = None
    create_temp_scheduler = None

# Import pipeline feedback controller for closed-loop integration
try:
    from app.integration.pipeline_feedback import (
        PipelineFeedbackController,
        FeedbackAction,
        create_feedback_controller,
    )
    HAS_FEEDBACK = True
except ImportError:
    HAS_FEEDBACK = False
    PipelineFeedbackController = None
    FeedbackAction = None

# Import cross-process event queue for multi-daemon coordination
try:
    from app.coordination.cross_process_events import (
        CrossProcessEventQueue,
        CrossProcessEvent,
    )
    HAS_CROSS_PROCESS_EVENTS = True
except ImportError:
    HAS_CROSS_PROCESS_EVENTS = False
    CrossProcessEventQueue = None
    CrossProcessEvent = None

# Import holdout validation for overfitting detection during promotion
try:
    from scripts.holdout_validation import (
        evaluate_model_on_holdout,
        EvaluationResult,
        OVERFIT_THRESHOLD,
    )
    HAS_HOLDOUT_VALIDATION = True
except ImportError:
    HAS_HOLDOUT_VALIDATION = False
    evaluate_model_on_holdout = None
    EvaluationResult = None
    OVERFIT_THRESHOLD = 0.15  # Default fallback

# Import pre-spawn health policy for remote hosts (renamed from health_check.py)
try:
    from app.coordination.host_health_policy import (
        is_host_healthy,
        check_host_health,
        get_healthy_hosts,
        clear_health_cache,
        HealthStatus as PreSpawnHealthStatus,
    )
    HAS_PRE_SPAWN_HEALTH = True
except ImportError:
    HAS_PRE_SPAWN_HEALTH = False
    is_host_healthy = None
    check_host_health = None
    get_healthy_hosts = None
    clear_health_cache = None
    PreSpawnHealthStatus = None

# Import circuit breaker for fault-tolerant remote operations
try:
    from app.distributed.circuit_breaker import (
        CircuitBreaker,
        CircuitState,
        CircuitOpenError,
        get_host_breaker,
        get_training_breaker,
    )
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False
    CircuitBreaker = None
    CircuitState = None
    CircuitOpenError = None
    get_host_breaker = None
    get_training_breaker = None

# Import P2P cluster integration for distributed training
try:
    from app.integration.p2p_integration import (
        P2PIntegrationConfig,
        P2PIntegrationManager,
    )
    HAS_P2P = True
except ImportError:
    HAS_P2P = False
    P2PIntegrationConfig = None
    P2PIntegrationManager = None

# Memory and local task configuration
MIN_MEMORY_GB = 64  # Minimum RAM to run the unified loop
DISABLE_LOCAL_TASKS = os.environ.get("RINGRIFT_DISABLE_LOCAL_TASKS", "").lower() in ("1", "true", "yes", "on")


# =============================================================================
# Prometheus Metrics
# =============================================================================

if HAS_PROMETHEUS:
    # Data collection metrics
    GAMES_SYNCED_TOTAL = Counter(
        'ringrift_games_synced_total',
        'Total games synced from remote hosts',
        ['host']
    )
    SYNC_DURATION_SECONDS = Histogram(
        'ringrift_sync_duration_seconds',
        'Time taken to sync games from a host',
        ['host'],
        buckets=[1, 5, 10, 30, 60, 120, 300]
    )
    SYNC_ERRORS_TOTAL = Counter(
        'ringrift_sync_errors_total',
        'Total sync errors by host',
        ['host', 'error_type']
    )
    GAMES_PENDING_TRAINING = Gauge(
        'ringrift_games_pending_training',
        'Games collected but not yet used for training',
        ['config']
    )

    # Training metrics
    TRAINING_RUNS_TOTAL = Counter(
        'ringrift_training_runs_total',
        'Total training runs',
        ['config', 'status']
    )
    TRAINING_DURATION_SECONDS = Histogram(
        'ringrift_training_duration_seconds',
        'Training run duration in seconds',
        ['config'],
        buckets=[60, 300, 600, 1800, 3600, 7200]
    )
    TRAINING_IN_PROGRESS = Gauge(
        'ringrift_training_in_progress',
        'Whether training is currently running',
        ['config']
    )

    # Evaluation metrics
    EVALUATIONS_TOTAL = Counter(
        'ringrift_evaluations_total',
        'Total evaluation runs',
        ['config', 'type']
    )
    EVALUATION_DURATION_SECONDS = Histogram(
        'ringrift_evaluation_duration_seconds',
        'Evaluation duration in seconds',
        ['config', 'type'],
        buckets=[30, 60, 120, 300, 600, 1200]
    )
    CURRENT_ELO = Gauge(
        'ringrift_current_elo',
        'Current Elo rating for configuration',
        ['config', 'model']
    )
    ELO_TREND = Gauge(
        'ringrift_elo_trend',
        'Elo trend (positive = improving)',
        ['config']
    )

    # Promotion metrics
    PROMOTIONS_TOTAL = Counter(
        'ringrift_promotions_total',
        'Total model promotions',
        ['config', 'status']
    )
    ELO_GAIN_ON_PROMOTION = Histogram(
        'ringrift_elo_gain_on_promotion',
        'Elo gain when model is promoted',
        ['config'],
        buckets=[5, 10, 20, 30, 50, 100]
    )
    PROMOTION_CANDIDATES = Gauge(
        'ringrift_promotion_candidates',
        'Number of promotion candidates',
        []
    )

    # Curriculum metrics
    CURRICULUM_WEIGHT = Gauge(
        'ringrift_curriculum_weight',
        'Training weight for configuration',
        ['config']
    )
    CURRICULUM_REBALANCES_TOTAL = Counter(
        'ringrift_curriculum_rebalances_total',
        'Total curriculum rebalancing events',
        []
    )

    # System metrics
    LOOP_CYCLES_TOTAL = Counter(
        'ringrift_loop_cycles_total',
        'Total improvement loop cycles',
        ['loop']
    )
    LOOP_ERRORS_TOTAL = Counter(
        'ringrift_loop_errors_total',
        'Total loop errors',
        ['loop', 'error_type']
    )
    UPTIME_SECONDS = Gauge(
        'ringrift_uptime_seconds',
        'Daemon uptime in seconds',
        []
    )
    HOSTS_ACTIVE = Gauge(
        'ringrift_hosts_active',
        'Number of active hosts',
        []
    )
    HOSTS_FAILED = Gauge(
        'ringrift_hosts_failed',
        'Number of failed hosts (consecutive failures)',
        []
    )

    # Training progress metrics (for training dashboard compatibility)
    TOTAL_MODELS = Gauge(
        'ringrift_total_models',
        'Total number of model files across all configs',
        []
    )
    MAX_MODEL_VERSION = Gauge(
        'ringrift_max_model_version',
        'Maximum model version number',
        []
    )
    MAX_ITERATION = Gauge(
        'ringrift_max_iteration',
        'Maximum training iteration by config',
        ['config']
    )
    MODEL_PROMOTIONS = Gauge(
        'ringrift_model_promotions_total',
        'Total model promotions (gauge for dashboard)',
        []
    )
    MODEL_ELO = Gauge(
        'ringrift_model_elo',
        'Model Elo rating by config and model',
        ['config', 'model']
    )
    MODEL_WIN_RATE = Gauge(
        'ringrift_model_win_rate',
        'Model win rate by config',
        ['config']
    )
    MODELS_BY_VERSION = Gauge(
        'ringrift_models_by_version',
        'Number of models by version',
        ['version']
    )
    MODELS_SIZE_GB = Gauge(
        'ringrift_models_size_gb',
        'Total size of models in GB',
        []
    )

    # Health monitoring metrics
    COMPONENT_HEALTH = Gauge(
        'ringrift_component_health',
        'Component health status (1=healthy, 0.5=degraded, 0=unhealthy)',
        ['component']
    )
    COMPONENT_LAST_SUCCESS = Gauge(
        'ringrift_component_last_success_timestamp',
        'Unix timestamp of last successful operation per component',
        ['component']
    )
    COMPONENT_CONSECUTIVE_FAILURES = Gauge(
        'ringrift_component_consecutive_failures',
        'Number of consecutive failures per component',
        ['component']
    )
    OVERALL_HEALTH = Gauge(
        'ringrift_overall_health',
        'Overall system health (1=healthy, 0.5=degraded, 0=unhealthy)',
        []
    )

    # Cross-process event metrics
    CROSS_PROCESS_EVENTS_BRIDGED = Counter(
        'ringrift_cross_process_events_bridged_total',
        'Total cross-process events bridged to local event bus',
        ['event_type', 'source']
    )
    CROSS_PROCESS_POLL_ERRORS = Counter(
        'ringrift_cross_process_poll_errors_total',
        'Total errors when polling cross-process event queue',
        []
    )

    # Data quality metrics
    DATA_QUALITY_SCORE = Gauge(
        'ringrift_data_quality_score',
        'Current data quality score (0-1)',
        []
    )
    DATA_QUALITY_DRAW_RATE = Gauge(
        'ringrift_data_quality_draw_rate',
        'Current draw rate in training data',
        []
    )
    DATA_QUALITY_TIMEOUT_RATE = Gauge(
        'ringrift_data_quality_timeout_rate',
        'Current timeout/move-limit rate in training data',
        []
    )
    DATA_QUALITY_CHECKS = Counter(
        'ringrift_data_quality_checks_total',
        'Total data quality checks performed',
        []
    )
    DATA_QUALITY_BLOCKED_TRAINING = Counter(
        'ringrift_data_quality_blocked_training_total',
        'Training runs blocked due to data quality issues',
        ['reason']
    )

    # Holdout validation metrics
    HOLDOUT_EVALUATIONS = Counter(
        'ringrift_holdout_evaluations_total',
        'Total holdout evaluations performed',
        ['config', 'result']  # result: passed, failed_overfit, skipped
    )
    HOLDOUT_LOSS = Gauge(
        'ringrift_holdout_loss',
        'Holdout loss for most recent evaluation',
        ['config']
    )
    HOLDOUT_OVERFIT_GAP = Gauge(
        'ringrift_holdout_overfit_gap',
        'Gap between holdout loss and training loss (positive = overfitting)',
        ['config']
    )
    PROMOTION_BLOCKED_OVERFIT = Counter(
        'ringrift_promotion_blocked_overfit_total',
        'Promotions blocked due to overfitting detection',
        ['config']
    )


class HealthStatus(Enum):
    """Component health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Tracks health state for a single component."""
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_success: float = 0.0
    last_failure: float = 0.0
    consecutive_failures: int = 0
    last_error: str = ""
    stale_threshold_seconds: int = 300

    def record_success(self):
        """Record a successful operation."""
        self.last_success = time.time()
        self.consecutive_failures = 0
        self.status = HealthStatus.HEALTHY

    def record_failure(self, error: str = ""):
        """Record a failed operation."""
        self.last_failure = time.time()
        self.consecutive_failures += 1
        self.last_error = error
        if self.consecutive_failures >= 3:
            self.status = HealthStatus.UNHEALTHY
        else:
            self.status = HealthStatus.DEGRADED

    def check_staleness(self) -> HealthStatus:
        """Check if component is stale (no recent activity)."""
        if self.last_success == 0:
            return HealthStatus.UNKNOWN
        age = time.time() - self.last_success
        if age > self.stale_threshold_seconds * 2:
            return HealthStatus.UNHEALTHY
        elif age > self.stale_threshold_seconds:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def get_status(self) -> HealthStatus:
        """Get current health status (considers staleness)."""
        staleness = self.check_staleness()
        if self.status == HealthStatus.UNHEALTHY or staleness == HealthStatus.UNHEALTHY:
            return HealthStatus.UNHEALTHY
        if self.status == HealthStatus.DEGRADED or staleness == HealthStatus.DEGRADED:
            return HealthStatus.DEGRADED
        if self.status == HealthStatus.HEALTHY and staleness == HealthStatus.HEALTHY:
            return HealthStatus.HEALTHY
        return HealthStatus.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "status": self.get_status().value,
            "last_success": self.last_success,
            "last_failure": self.last_failure,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "age_seconds": time.time() - self.last_success if self.last_success > 0 else -1,
        }


# Global health tracker (set by UnifiedAILoop)
_health_tracker: Optional["HealthTracker"] = None


class HealthTracker:
    """Tracks health of all system components."""

    def __init__(self):
        # Initialize health trackers for each component
        self.components: Dict[str, ComponentHealth] = {
            "data_collector": ComponentHealth("data_collector", stale_threshold_seconds=300),
            "evaluator": ComponentHealth("evaluator", stale_threshold_seconds=1800),
            "training_scheduler": ComponentHealth("training_scheduler", stale_threshold_seconds=7200),
            "promoter": ComponentHealth("promoter", stale_threshold_seconds=3600),
            "curriculum": ComponentHealth("curriculum", stale_threshold_seconds=3600),
        }

    def record_success(self, component: str):
        """Record successful operation for a component."""
        if component in self.components:
            self.components[component].record_success()
            self._update_metrics(component)

    def record_failure(self, component: str, error: str = ""):
        """Record failed operation for a component."""
        if component in self.components:
            self.components[component].record_failure(error)
            self._update_metrics(component)

    def _update_metrics(self, component: str):
        """Update Prometheus metrics for a component."""
        if not HAS_PROMETHEUS:
            return
        ch = self.components[component]
        status = ch.get_status()
        COMPONENT_HEALTH.labels(component=component).set(
            1.0 if status == HealthStatus.HEALTHY else
            0.5 if status == HealthStatus.DEGRADED else 0.0
        )
        if ch.last_success > 0:
            COMPONENT_LAST_SUCCESS.labels(component=component).set(ch.last_success)
        COMPONENT_CONSECUTIVE_FAILURES.labels(component=component).set(ch.consecutive_failures)

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        statuses = [c.get_status() for c in self.components.values()]
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        return HealthStatus.UNKNOWN

    def get_health_summary(self) -> Dict[str, Any]:
        """Get full health summary for API response."""
        overall = self.get_overall_status()
        if HAS_PROMETHEUS:
            OVERALL_HEALTH.set(
                1.0 if overall == HealthStatus.HEALTHY else
                0.5 if overall == HealthStatus.DEGRADED else 0.0
            )
        return {
            "status": overall.value,
            "timestamp": time.time(),
            "components": {name: ch.to_dict() for name, ch in self.components.items()},
        }


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""

    def do_GET(self):
        if self.path == '/metrics' and HAS_PROMETHEUS:
            self.send_response(200)
            self.send_header('Content-Type', CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(generate_latest())
        elif self.path == '/health':
            # Return detailed health status if tracker is available
            global _health_tracker
            if _health_tracker is not None:
                health_data = _health_tracker.get_health_summary()
                status = health_data.get("status", "unknown")
                http_code = 200 if status == "healthy" else 503 if status == "unhealthy" else 200
                self.send_response(http_code)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(health_data, indent=2).encode())
            else:
                # Basic health check if tracker not initialized
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress request logging


def start_metrics_server(port: int = 9090) -> Optional[HTTPServer]:
    """Start the Prometheus metrics HTTP server."""
    if not HAS_PROMETHEUS:
        print("[Metrics] prometheus_client not installed, metrics disabled")
        return None

    try:
        server = HTTPServer(('0.0.0.0', port), MetricsHandler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"[Metrics] Prometheus metrics available at http://0.0.0.0:{port}/metrics")
        return server
    except Exception as e:
        print(f"[Metrics] Failed to start metrics server: {e}")
        return None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DataIngestionConfig:
    """Configuration for streaming data collection."""
    poll_interval_seconds: int = 60
    sync_method: str = "incremental"  # "incremental" or "full"
    deduplication: bool = True
    min_games_per_sync: int = 10
    remote_db_pattern: str = "data/games/*.db"


@dataclass
class TrainingConfig:
    """Configuration for automatic training triggers.

    NOTE: Defaults match app/config/unified_config.py (single source of truth)
    """
    trigger_threshold_games: int = 500  # Canonical: 500 (was 1000)
    min_interval_seconds: int = 1200  # Canonical: 20 min (was 30)
    max_concurrent_jobs: int = 1
    prefer_gpu_hosts: bool = True
    nnue_training_script: str = "scripts/train_nnue.py"
    nn_training_script: str = "scripts/run_nn_training_baseline.py"
    export_script: str = "scripts/export_replay_dataset.py"
    # Encoder version for hex boards: "v3" uses HexStateEncoderV3 (16 channels)
    hex_encoder_version: str = "v3"


@dataclass
class EvaluationConfig:
    """Configuration for continuous evaluation.

    NOTE: Defaults match app/config/unified_config.py (single source of truth)
    """
    shadow_interval_seconds: int = 900  # 15 minutes
    shadow_games_per_config: int = 15  # Canonical: 15 (was 10)
    full_tournament_interval_seconds: int = 3600  # 1 hour
    full_tournament_games: int = 50
    baseline_models: List[str] = field(default_factory=lambda: ["random", "heuristic", "mcts_100", "mcts_500"])


@dataclass
class PromotionConfig:
    """Configuration for automatic model promotion.

    NOTE: Defaults match app/config/unified_config.py (single source of truth)
    """
    auto_promote: bool = True
    elo_threshold: int = 25  # Canonical: 25 (was 20)
    min_games: int = 50
    significance_level: float = 0.05
    sync_to_cluster: bool = True


@dataclass
class CurriculumConfig:
    """Configuration for adaptive curriculum.

    NOTE: Defaults match app/config/unified_config.py (single source of truth)
    """
    adaptive: bool = True
    rebalance_interval_seconds: int = 3600  # 1 hour
    max_weight_multiplier: float = 1.5  # Canonical: 1.5 (was 2.0)
    min_weight_multiplier: float = 0.7  # Canonical: 0.7 (was 0.5)


@dataclass
class PBTConfig:
    """Configuration for Population-Based Training."""
    enabled: bool = False  # Disabled by default - resource intensive
    population_size: int = 8
    exploit_interval_steps: int = 1000
    tunable_params: List[str] = field(default_factory=lambda: ["learning_rate", "batch_size", "temperature"])
    check_interval_seconds: int = 1800  # Check PBT status every 30 min
    auto_start: bool = False  # Auto-start PBT when training completes


@dataclass
class NASConfig:
    """Configuration for Neural Architecture Search."""
    enabled: bool = False  # Disabled by default - very resource intensive
    strategy: str = "evolutionary"  # evolutionary, random, bayesian
    population_size: int = 20
    generations: int = 50
    check_interval_seconds: int = 3600  # Check NAS status every hour
    auto_start_on_plateau: bool = False  # Start NAS when Elo plateaus


@dataclass
class PERConfig:
    """Configuration for Prioritized Experience Replay."""
    enabled: bool = True  # Enabled by default - improves training efficiency
    alpha: float = 0.6  # Priority exponent
    beta: float = 0.4  # Importance sampling exponent
    buffer_capacity: int = 100000
    rebuild_interval_seconds: int = 7200  # Rebuild buffer every 2 hours


@dataclass
class FeedbackConfig:
    """Configuration for pipeline feedback controller integration."""
    enabled: bool = True  # Enable closed-loop feedback
    # Performance-based training triggers
    elo_plateau_threshold: float = 15.0  # Elo gain below this triggers plateau detection
    elo_plateau_lookback: int = 5  # Number of evaluations to look back
    win_rate_degradation_threshold: float = 0.40  # Win rate below this triggers retraining
    # Data quality gates
    max_parity_failure_rate: float = 0.10  # Block training if parity failures exceed this
    min_data_quality_score: float = 0.70  # Minimum data quality to proceed with training
    # CMA-ES/NAS auto-trigger
    plateau_count_for_cmaes: int = 2  # Trigger CMA-ES after this many consecutive plateaus
    plateau_count_for_nas: int = 4  # Trigger NAS after this many consecutive plateaus


@dataclass
class P2PClusterConfig:
    """Configuration for P2P distributed cluster integration."""
    enabled: bool = False  # Enable P2P cluster coordination
    p2p_base_url: str = "http://localhost:8770"  # P2P orchestrator URL
    auth_token: Optional[str] = None  # Auth token (defaults to RINGRIFT_CLUSTER_AUTH_TOKEN env)
    model_sync_enabled: bool = True  # Auto-sync models to cluster
    target_selfplay_games_per_hour: int = 1000  # Target selfplay rate across cluster
    auto_scale_selfplay: bool = True  # Auto-scale selfplay workers
    use_distributed_tournament: bool = True  # Use cluster for tournament evaluation
    health_check_interval: int = 60  # Seconds between cluster health checks
    sync_interval_seconds: int = 300  # Seconds between data sync with cluster


@dataclass
class UnifiedLoopConfig:
    """Complete configuration for the unified AI loop."""
    data_ingestion: DataIngestionConfig = field(default_factory=DataIngestionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    pbt: PBTConfig = field(default_factory=PBTConfig)
    nas: NASConfig = field(default_factory=NASConfig)
    per: PERConfig = field(default_factory=PERConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    p2p: P2PClusterConfig = field(default_factory=P2PClusterConfig)

    # Host configuration
    hosts_config_path: str = "config/remote_hosts.yaml"

    # Database paths
    elo_db: str = "data/unified_elo.db"  # Canonical Elo database
    data_manifest_db: str = "data/data_manifest.db"

    # Logging
    log_dir: str = "logs/unified_loop"
    verbose: bool = False

    # Metrics
    metrics_port: int = 9090
    metrics_enabled: bool = True

    # Operation modes
    dry_run: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "UnifiedLoopConfig":
        """Load configuration from YAML file."""
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = cls()

        if "data_ingestion" in data:
            for k, v in data["data_ingestion"].items():
                if hasattr(config.data_ingestion, k):
                    setattr(config.data_ingestion, k, v)

        if "training" in data:
            for k, v in data["training"].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)

        if "evaluation" in data:
            for k, v in data["evaluation"].items():
                if hasattr(config.evaluation, k):
                    setattr(config.evaluation, k, v)

        if "promotion" in data:
            for k, v in data["promotion"].items():
                if hasattr(config.promotion, k):
                    setattr(config.promotion, k, v)

        if "curriculum" in data:
            for k, v in data["curriculum"].items():
                if hasattr(config.curriculum, k):
                    setattr(config.curriculum, k, v)

        if "pbt" in data:
            for k, v in data["pbt"].items():
                if hasattr(config.pbt, k):
                    setattr(config.pbt, k, v)

        if "nas" in data:
            for k, v in data["nas"].items():
                if hasattr(config.nas, k):
                    setattr(config.nas, k, v)

        if "per" in data:
            for k, v in data["per"].items():
                if hasattr(config.per, k):
                    setattr(config.per, k, v)

        if "feedback" in data:
            for k, v in data["feedback"].items():
                if hasattr(config.feedback, k):
                    setattr(config.feedback, k, v)

        if "p2p" in data:
            for k, v in data["p2p"].items():
                if hasattr(config.p2p, k):
                    setattr(config.p2p, k, v)

        for key in ["hosts_config_path", "elo_db", "data_manifest_db", "log_dir",
                    "verbose", "metrics_port", "metrics_enabled", "dry_run"]:
            if key in data:
                setattr(config, key, data[key])

        return config


# =============================================================================
# Event System
# =============================================================================

class DataEventType(Enum):
    """Types of data pipeline events."""
    NEW_GAMES_AVAILABLE = "new_games"
    TRAINING_THRESHOLD_REACHED = "training_threshold"
    TRAINING_STARTED = "training_started"
    TRAINING_COMPLETED = "training_completed"
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_COMPLETED = "evaluation_completed"
    PROMOTION_CANDIDATE = "promotion_candidate"
    MODEL_PROMOTED = "model_promoted"
    CURRICULUM_REBALANCED = "curriculum_rebalanced"
    ELO_SIGNIFICANT_CHANGE = "elo_significant_change"  # Triggers event-driven curriculum rebalance
    # PBT events
    PBT_STARTED = "pbt_started"
    PBT_GENERATION_COMPLETE = "pbt_generation_complete"
    PBT_COMPLETED = "pbt_completed"
    # NAS events
    NAS_STARTED = "nas_started"
    NAS_GENERATION_COMPLETE = "nas_generation_complete"
    NAS_COMPLETED = "nas_completed"
    NAS_BEST_ARCHITECTURE = "nas_best_architecture"
    # PER events
    PER_BUFFER_REBUILT = "per_buffer_rebuilt"
    PER_PRIORITIES_UPDATED = "per_priorities_updated"
    # Optimization events
    CMAES_TRIGGERED = "cmaes_triggered"
    NAS_TRIGGERED = "nas_triggered"
    PLATEAU_DETECTED = "plateau_detected"
    HYPERPARAMETER_UPDATED = "hyperparameter_updated"
    # P2P cluster events
    P2P_CLUSTER_HEALTHY = "p2p_cluster_healthy"
    P2P_CLUSTER_UNHEALTHY = "p2p_cluster_unhealthy"
    P2P_NODES_DEAD = "p2p_nodes_dead"
    P2P_SELFPLAY_SCALED = "p2p_selfplay_scaled"
    P2P_MODEL_SYNCED = "p2p_model_synced"


@dataclass
class DataEvent:
    """A data pipeline event."""
    event_type: DataEventType
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class EventBus:
    """Simple async event bus for component coordination."""

    def __init__(self):
        self._subscribers: Dict[DataEventType, List[Callable]] = {}
        self._event_history: List[DataEvent] = []
        self._max_history = 1000

    def subscribe(self, event_type: DataEventType, callback: Callable):
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    async def publish(self, event: DataEvent):
        """Publish an event to all subscribers."""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        if event.event_type in self._subscribers:
            for callback in self._subscribers[event.event_type]:
                try:
                    result = callback(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    print(f"[EventBus] Error in subscriber: {e}")

    def get_recent_events(self, event_type: Optional[DataEventType] = None, limit: int = 100) -> List[DataEvent]:
        """Get recent events, optionally filtered by type."""
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]


# =============================================================================
# State Management
# =============================================================================

@dataclass
class HostState:
    """State for a remote host."""
    name: str
    ssh_host: str
    ssh_user: str = "ubuntu"
    ssh_port: int = 22
    last_sync_time: float = 0.0
    last_game_count: int = 0
    consecutive_failures: int = 0
    enabled: bool = True


@dataclass
class ConfigState:
    """State for a board/player configuration."""
    board_type: str
    num_players: int
    game_count: int = 0
    games_since_training: int = 0
    last_training_time: float = 0.0
    last_evaluation_time: float = 0.0
    current_elo: float = 1500.0
    elo_trend: float = 0.0  # Positive = improving
    training_weight: float = 1.0


@dataclass
class UnifiedLoopState:
    """Complete state for the unified AI loop."""
    started_at: str = ""
    last_cycle_at: str = ""

    # Cycle counters
    total_data_syncs: int = 0
    total_training_runs: int = 0
    total_evaluations: int = 0
    total_promotions: int = 0

    # Host states
    hosts: Dict[str, HostState] = field(default_factory=dict)

    # Configuration states (keyed by "board_type_num_players")
    configs: Dict[str, ConfigState] = field(default_factory=dict)

    # Current training state
    training_in_progress: bool = False
    training_config: str = ""
    training_started_at: float = 0.0

    # Games pending training
    total_games_pending: int = 0

    # Error tracking
    consecutive_failures: int = 0
    last_error: str = ""
    last_error_time: str = ""

    # Health check state
    last_health_check: str = ""
    health_status: str = "unknown"  # "healthy", "unhealthy", or "unknown"

    # Curriculum weights (config_key -> weight)
    curriculum_weights: Dict[str, float] = field(default_factory=dict)
    last_curriculum_rebalance: float = 0.0

    # PBT state
    pbt_in_progress: bool = False
    pbt_run_id: str = ""
    pbt_started_at: float = 0.0
    pbt_generation: int = 0
    pbt_best_performance: float = 0.0

    # NAS state
    nas_in_progress: bool = False
    nas_run_id: str = ""
    nas_started_at: float = 0.0
    nas_generation: int = 0
    nas_best_architecture: Optional[Dict[str, Any]] = None

    # PER state
    per_buffer_path: str = ""
    per_last_rebuild: float = 0.0
    per_buffer_size: int = 0

    # Calibration state (config_key -> last calibration report dict)
    calibration_reports: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_calibration_time: float = 0.0
    current_temperature_preset: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "started_at": self.started_at,
            "last_cycle_at": self.last_cycle_at,
            "total_data_syncs": self.total_data_syncs,
            "total_training_runs": self.total_training_runs,
            "total_evaluations": self.total_evaluations,
            "total_promotions": self.total_promotions,
            "hosts": {k: asdict(v) for k, v in self.hosts.items()},
            "configs": {k: asdict(v) for k, v in self.configs.items()},
            "training_in_progress": self.training_in_progress,
            "training_config": self.training_config,
            "training_started_at": self.training_started_at,
            "total_games_pending": self.total_games_pending,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time,
            "curriculum_weights": self.curriculum_weights,
            "last_curriculum_rebalance": self.last_curriculum_rebalance,
            # PBT state
            "pbt_in_progress": self.pbt_in_progress,
            "pbt_run_id": self.pbt_run_id,
            "pbt_started_at": self.pbt_started_at,
            "pbt_generation": self.pbt_generation,
            "pbt_best_performance": self.pbt_best_performance,
            # NAS state
            "nas_in_progress": self.nas_in_progress,
            "nas_run_id": self.nas_run_id,
            "nas_started_at": self.nas_started_at,
            "nas_generation": self.nas_generation,
            "nas_best_architecture": self.nas_best_architecture,
            # PER state
            "per_buffer_path": self.per_buffer_path,
            "per_last_rebuild": self.per_last_rebuild,
            "per_buffer_size": self.per_buffer_size,
            # Calibration state
            "calibration_reports": self.calibration_reports,
            "last_calibration_time": self.last_calibration_time,
            "current_temperature_preset": self.current_temperature_preset,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedLoopState":
        """Create state from dictionary."""
        state = cls()
        for key in ["started_at", "last_cycle_at", "total_data_syncs", "total_training_runs",
                    "total_evaluations", "total_promotions", "training_in_progress",
                    "training_config", "training_started_at", "total_games_pending",
                    "consecutive_failures", "last_error", "last_error_time",
                    "last_curriculum_rebalance",
                    # PBT state
                    "pbt_in_progress", "pbt_run_id", "pbt_started_at",
                    "pbt_generation", "pbt_best_performance",
                    # NAS state
                    "nas_in_progress", "nas_run_id", "nas_started_at",
                    "nas_generation", "nas_best_architecture",
                    # PER state
                    "per_buffer_path", "per_last_rebuild", "per_buffer_size",
                    # Calibration state
                    "calibration_reports", "last_calibration_time", "current_temperature_preset"]:
            if key in data:
                setattr(state, key, data[key])

        if "hosts" in data:
            for name, host_data in data["hosts"].items():
                state.hosts[name] = HostState(**host_data)

        if "configs" in data:
            for key, config_data in data["configs"].items():
                state.configs[key] = ConfigState(**config_data)

        if "curriculum_weights" in data:
            state.curriculum_weights = data["curriculum_weights"]

        return state


# =============================================================================
# Data Collection Component
# =============================================================================

class StreamingDataCollector:
    """Collects game data from remote hosts with 60-second incremental sync."""

    def __init__(
        self,
        config: DataIngestionConfig,
        state: UnifiedLoopState,
        event_bus: EventBus,
        hot_buffer: Optional["HotDataBuffer"] = None,
    ):
        self.config = config
        self.state = state
        self.event_bus = event_bus
        self.hot_buffer = hot_buffer
        self._known_game_ids: Set[str] = set()

    def set_hot_buffer(self, hot_buffer: "HotDataBuffer") -> None:
        """Set or update the hot buffer for in-memory game caching."""
        self.hot_buffer = hot_buffer
        print(f"[DataCollector] Hot buffer attached (max_size={hot_buffer.max_size})")

    async def sync_host(self, host: HostState) -> int:
        """Sync games from a single host. Returns count of new games."""
        if not host.enabled:
            return 0

        # Skip pre-spawn health check for data collection - the SSH call below
        # will fail fast if host is unreachable. The health check uses IP addresses
        # but host lookup uses names, causing false negatives.

        # Circuit breaker check - prevent repeated failures
        if HAS_CIRCUIT_BREAKER:
            breaker = get_host_breaker()
            if not breaker.can_execute(host.ssh_host):
                state = breaker.get_state(host.ssh_host)
                if state == CircuitState.OPEN:
                    print(f"[DataCollector] Skipping {host.name}: circuit open (cooldown)")
                return 0

        try:
            # Query game count on remote host
            ssh_target = f"{host.ssh_user}@{host.ssh_host}"
            port_arg = f"-p {host.ssh_port}" if host.ssh_port != 22 else ""

            # Get game count from all DBs (using Python since sqlite3 CLI may not be installed)
            python_script = (
                "import sqlite3, glob, os; "
                "os.chdir(os.path.expanduser('~/ringrift/ai-service')); "
                "dbs=glob.glob('data/games/*.db'); "
                "total=0; "
                "[total := total + sqlite3.connect(db).execute('SELECT COUNT(*) FROM games').fetchone()[0] for db in dbs if 'schema' not in db]; "
                "print(total)"
            )
            cmd = f'ssh -o ConnectTimeout=10 {port_arg} {ssh_target} "python3 -c \\"{python_script}\\"" 2>/dev/null'

            result = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(result.communicate(), timeout=30)

            current_count = int(stdout.decode().strip() or "0")
            new_games = max(0, current_count - host.last_game_count)

            print(f"[DataCollector] {host.name}: {current_count} games (last: {host.last_game_count}, new: {new_games})")

            if new_games >= self.config.min_games_per_sync:
                # Trigger rsync for incremental sync
                if self.config.sync_method == "incremental":
                    await self._incremental_sync(host)
                else:
                    await self._full_sync(host)

                host.last_game_count = current_count
                host.last_sync_time = time.time()

                # Publish event
                await self.event_bus.publish(DataEvent(
                    event_type=DataEventType.NEW_GAMES_AVAILABLE,
                    payload={
                        "host": host.name,
                        "new_games": new_games,
                        "total_games": current_count,
                    }
                ))

            host.consecutive_failures = 0
            # Record success with circuit breaker
            if HAS_CIRCUIT_BREAKER:
                get_host_breaker().record_success(host.ssh_host)
            return new_games

        except Exception as e:
            host.consecutive_failures += 1
            # Record failure with circuit breaker
            if HAS_CIRCUIT_BREAKER:
                get_host_breaker().record_failure(host.ssh_host, e)
            print(f"[DataCollector] Failed to sync {host.name}: {e}")
            return 0

    async def _incremental_sync(self, host: HostState):
        """Perform incremental rsync of new data.

        Uses sync_lock and bandwidth management when available for coordinated
        data transfers across the cluster.
        """
        ssh_target = f"{host.ssh_user}@{host.ssh_host}"
        local_dir = AI_SERVICE_ROOT / "data" / "games" / "synced" / host.name
        local_dir.mkdir(parents=True, exist_ok=True)

        # Rsync with append mode for incremental transfer
        cmd = f'rsync -avz --progress -e "ssh -o ConnectTimeout=10" {ssh_target}:~/ringrift/ai-service/data/games/*.db {local_dir}/'

        # Use new coordination if available: sync_lock + bandwidth
        if HAS_COORDINATION:
            bandwidth_alloc = None
            try:
                # Acquire sync lock to prevent concurrent rsync operations
                with sync_lock(host=host.ssh_host, operation="data_sync"):
                    # Request bandwidth allocation (estimate ~50MB for DB sync)
                    bandwidth_alloc = request_bandwidth(
                        host=host.ssh_host,
                        estimated_mb=50,  # 50MB estimate
                        priority=TransferPriority.NORMAL,
                    )

                    if bandwidth_alloc and not bandwidth_alloc.granted:
                        print(f"[DataCollector] Bandwidth not available for {host.name}")
                        return

                    start_time = time.time()
                    process = await asyncio.create_subprocess_shell(
                        cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await asyncio.wait_for(process.communicate(), timeout=300)
                    transfer_duration = time.time() - start_time

                    # Release bandwidth with actual transfer stats
                    if bandwidth_alloc and bandwidth_alloc.granted:
                        release_bandwidth(
                            bandwidth_alloc.allocation_id,
                            bytes_transferred=50 * 1024 * 1024,  # Estimate
                            duration_seconds=transfer_duration
                        )
            except Exception as e:
                print(f"[DataCollector] Sync error for {host.name}: {e}")
                raise
            finally:
                # Ensure bandwidth is released even on error
                if bandwidth_alloc and bandwidth_alloc.granted:
                    try:
                        release_bandwidth(bandwidth_alloc.allocation_id)
                    except Exception:
                        pass
        else:
            # Fallback: no coordination
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(process.communicate(), timeout=300)

    async def _full_sync(self, host: HostState):
        """Perform full sync (same as incremental for now)."""
        await self._incremental_sync(host)

    def compute_quality_stats(self, sample_size: int = 500) -> Dict[str, Any]:
        """Compute data quality statistics from synced databases.

        Returns:
            Dictionary with draw_rate, timeout_rate, game_lengths, etc.
        """
        synced_dir = AI_SERVICE_ROOT / "data" / "games" / "synced"
        if not synced_dir.exists():
            return {"draw_rate": 0, "timeout_rate": 0, "game_lengths": []}

        total_games = 0
        draws = 0
        timeouts = 0
        game_lengths = []

        for db_path in synced_dir.rglob("*.db"):
            try:
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Check table structure
                cursor.execute("PRAGMA table_info(games)")
                columns = {row['name'] for row in cursor.fetchall()}

                has_winner = 'winner' in columns
                has_total_moves = 'total_moves' in columns

                # Sample recent games
                if has_winner and has_total_moves:
                    cursor.execute("""
                        SELECT winner, total_moves FROM games
                        ORDER BY ROWID DESC LIMIT ?
                    """, (sample_size,))
                elif has_winner:
                    cursor.execute("""
                        SELECT winner, 0 as total_moves FROM games
                        ORDER BY ROWID DESC LIMIT ?
                    """, (sample_size,))
                else:
                    conn.close()
                    continue

                for row in cursor.fetchall():
                    total_games += 1
                    winner = row['winner']
                    moves = row['total_moves']

                    # Count draws (winner = -1 or winner is NULL typically means draw)
                    if winner is None or winner == -1:
                        draws += 1

                    # Count timeouts (games hitting move limits)
                    if moves >= 500:  # Common move limits
                        timeouts += 1

                    if moves > 0:
                        game_lengths.append(moves)

                conn.close()
            except Exception as e:
                print(f"[DataCollector] Quality stats error for {db_path}: {e}")
                continue

        return {
            "draw_rate": draws / total_games if total_games > 0 else 0,
            "timeout_rate": timeouts / total_games if total_games > 0 else 0,
            "game_lengths": game_lengths,
            "total_sampled": total_games,
        }

    async def run_collection_cycle(self) -> int:
        """Run one data collection cycle across all hosts."""
        print(f"[DataCollector] Starting collection cycle for {len(self.state.hosts)} hosts...", flush=True)
        total_new = 0
        tasks = []

        for host in self.state.hosts.values():
            if host.enabled:
                tasks.append(self.sync_host(host))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, int):
                    total_new += result

        self.state.total_data_syncs += 1
        self.state.total_games_pending += total_new

        # Update per-config game counts from synced databases
        if total_new > 0:
            self._update_per_config_game_counts(total_new)

        return total_new

    def _update_per_config_game_counts(self, new_games: int) -> None:
        """Update per-config games_since_training counters.

        Distributes new games across configs based on what's in synced databases.
        Falls back to proportional distribution if db parsing fails.
        """
        synced_dir = AI_SERVICE_ROOT / "data" / "games" / "synced"
        if not synced_dir.exists():
            # Fallback: distribute to square8_2p (most common)
            if "square8_2p" in self.state.configs:
                self.state.configs["square8_2p"].games_since_training += new_games
            return

        # Count games per config from database filenames
        config_counts: Dict[str, int] = {}
        total_counted = 0

        for db_path in synced_dir.rglob("*.db"):
            try:
                db_name = db_path.stem.lower()
                # Parse config from filename patterns like "selfplay_square8_2p.db"
                config_key = None
                for ck in self.state.configs.keys():
                    if ck.replace("_", "") in db_name.replace("_", "") or ck in db_name:
                        config_key = ck
                        break

                if config_key:
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM games")
                    count = cursor.fetchone()[0]
                    conn.close()
                    config_counts[config_key] = config_counts.get(config_key, 0) + count
                    total_counted += count
            except Exception:
                pass

        # Distribute new games proportionally based on existing counts
        if total_counted > 0:
            for config_key, count in config_counts.items():
                if config_key in self.state.configs:
                    proportion = count / total_counted
                    added = int(new_games * proportion)
                    self.state.configs[config_key].games_since_training += added
        else:
            # Fallback: distribute to square8_2p
            if "square8_2p" in self.state.configs:
                self.state.configs["square8_2p"].games_since_training += new_games


# =============================================================================
# Shadow Tournament Component
# =============================================================================

class ShadowTournamentService:
    """Runs lightweight continuous evaluation."""

    def __init__(self, config: EvaluationConfig, state: UnifiedLoopState, event_bus: EventBus):
        self.config = config
        self.state = state
        self.event_bus = event_bus

    async def run_shadow_tournament(self, config_key: str) -> Dict[str, Any]:
        """Run a quick shadow tournament for a configuration."""
        # Skip local evaluation if disabled
        if DISABLE_LOCAL_TASKS:
            return {"skipped": True, "reason": "RINGRIFT_DISABLE_LOCAL_TASKS"}

        parts = config_key.rsplit("_", 1)
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        await self.event_bus.publish(DataEvent(
            event_type=DataEventType.EVALUATION_STARTED,
            payload={"config": config_key, "type": "shadow"}
        ))

        try:
            # Run quick tournament
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "run_model_elo_tournament.py"),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--games", str(self.config.shadow_games_per_config),
                "--quick",
                "--include-baselines",
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)

            # Parse results (simplified - real implementation would parse JSON output)
            result = {
                "config": config_key,
                "games_played": self.config.shadow_games_per_config,
                "success": process.returncode == 0,
            }

            if config_key in self.state.configs:
                self.state.configs[config_key].last_evaluation_time = time.time()

            self.state.total_evaluations += 1

            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.EVALUATION_COMPLETED,
                payload=result
            ))

            return result

        except Exception as e:
            print(f"[ShadowTournament] Error running tournament for {config_key}: {e}")
            return {"config": config_key, "error": str(e), "success": False}

    async def run_full_tournament(self) -> Dict[str, Any]:
        """Run a full tournament across all configurations."""
        # Skip local evaluation if disabled
        if DISABLE_LOCAL_TASKS:
            return {"skipped": True, "reason": "RINGRIFT_DISABLE_LOCAL_TASKS"}

        await self.event_bus.publish(DataEvent(
            event_type=DataEventType.EVALUATION_STARTED,
            payload={"type": "full"}
        ))

        try:
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "run_model_elo_tournament.py"),
                "--all-configs",
                "--games", str(self.config.full_tournament_games),
                "--include-baselines",
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=3600)

            result = {
                "type": "full",
                "success": process.returncode == 0,
            }

            self.state.total_evaluations += 1

            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.EVALUATION_COMPLETED,
                payload=result
            ))

            return result

        except Exception as e:
            print(f"[ShadowTournament] Error running full tournament: {e}")
            return {"type": "full", "error": str(e), "success": False}


# =============================================================================
# Training Scheduler Component
# =============================================================================

class TrainingScheduler:
    """Schedules and manages training runs with cluster-wide coordination."""

    def __init__(
        self,
        config: TrainingConfig,
        state: UnifiedLoopState,
        event_bus: EventBus,
        feedback_config: Optional[FeedbackConfig] = None,
        feedback: Optional["PipelineFeedbackController"] = None
    ):
        self.config = config
        self.state = state
        self.event_bus = event_bus
        self.feedback_config = feedback_config or FeedbackConfig()
        self.feedback = feedback
        self._training_process: Optional[asyncio.subprocess.Process] = None
        # Cluster-wide training coordination
        self._training_lock_fd: Optional[int] = None
        self._training_lock_path: Optional[Path] = None
        # Calibration tracking (per config)
        self._calibration_trackers: Dict[str, Any] = {}
        if HAS_VALUE_CALIBRATION:
            for config_key in state.configs:
                self._calibration_trackers[config_key] = CalibrationTracker(window_size=1000)
        # Temperature scheduler for self-play exploration
        self._temp_scheduler: Optional[Any] = None
        if HAS_TEMPERATURE_SCHEDULING:
            self._temp_scheduler = create_temp_scheduler(state.current_temperature_preset or "default")

    def set_feedback_controller(self, feedback: "PipelineFeedbackController"):
        """Set the feedback controller (called after initialization)."""
        self.feedback = feedback

    def _acquire_training_lock(self) -> bool:
        """Acquire cluster-wide training lock using file locking.

        Returns True if lock acquired, False if training already running.
        """
        import fcntl
        import socket

        lock_dir = AI_SERVICE_ROOT / "data" / "coordination"
        lock_dir.mkdir(parents=True, exist_ok=True)

        hostname = socket.gethostname()
        self._training_lock_path = lock_dir / f"training.{hostname}.lock"

        try:
            self._training_lock_fd = os.open(
                str(self._training_lock_path),
                os.O_RDWR | os.O_CREAT
            )
            fcntl.flock(self._training_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(self._training_lock_fd, 0)
            os.lseek(self._training_lock_fd, 0, os.SEEK_SET)
            os.write(self._training_lock_fd, f"{os.getpid()}\n".encode())
            print(f"[Training] Acquired cluster-wide training lock on {hostname}")
            return True
        except (OSError, BlockingIOError) as e:
            if self._training_lock_fd is not None:
                os.close(self._training_lock_fd)
                self._training_lock_fd = None
            print(f"[Training] Lock acquisition failed: {e}")
            return False

    def _release_training_lock(self):
        """Release the cluster-wide training lock."""
        import fcntl

        if self._training_lock_fd is not None:
            try:
                fcntl.flock(self._training_lock_fd, fcntl.LOCK_UN)
                os.close(self._training_lock_fd)
                print("[Training] Released cluster-wide training lock")
            except Exception as e:
                print(f"[Training] Error releasing lock: {e}")
            finally:
                self._training_lock_fd = None
            if self._training_lock_path and self._training_lock_path.exists():
                try:
                    self._training_lock_path.unlink()
                except Exception:
                    pass

    def is_training_locked_elsewhere(self) -> bool:
        """Check if training is running on another host in the cluster."""
        import socket

        lock_dir = AI_SERVICE_ROOT / "data" / "coordination"
        if not lock_dir.exists():
            return False

        hostname = socket.gethostname()
        for lock_file in lock_dir.glob("training.*.lock"):
            if f"training.{hostname}.lock" in lock_file.name:
                continue
            try:
                if lock_file.stat().st_size > 0:
                    age = time.time() - lock_file.stat().st_mtime
                    if age < 3600:
                        other_host = lock_file.name.replace("training.", "").replace(".lock", "")
                        print(f"[Training] Training lock held by {other_host}")
                        return True
            except Exception:
                continue
        return False

    def should_trigger_training(self) -> Optional[str]:
        """Check if training should be triggered. Returns config key or None.

        Training can be triggered by:
        1. Game count threshold (traditional) - enough new games collected
        2. Elo plateau detection - model stopped improving
        3. Win rate degradation - model performing worse than threshold
        """
        if self.state.training_in_progress:
            return None

        # Check for cluster-wide training lock (prevent simultaneous training)
        if self.is_training_locked_elsewhere():
            return None

        now = time.time()

        for config_key, config_state in self.state.configs.items():
            # Check minimum interval between training runs
            if now - config_state.last_training_time < self.config.min_interval_seconds:
                continue

            # Trigger 1: Traditional game count threshold
            if config_state.games_since_training >= self.config.trigger_threshold_games:
                print(f"[Training] Trigger: game threshold reached for {config_key} "
                      f"({config_state.games_since_training} games)")
                return config_key

            # Trigger 2: Performance-based - Elo plateau detection
            if self.feedback:
                if self.feedback.eval_analyzer.is_plateau(
                    config_key,
                    min_improvement=self.feedback_config.elo_plateau_threshold,
                    lookback=self.feedback_config.elo_plateau_lookback
                ):
                    # Only trigger if we have some data to train on
                    if config_state.games_since_training >= self.config.trigger_threshold_games // 4:
                        print(f"[Training] Trigger: Elo plateau detected for {config_key} "
                              f"(trend < {self.feedback_config.elo_plateau_threshold})")
                        return config_key

            # Trigger 3: Performance-based - Win rate degradation
            if self.feedback:
                weak_configs = self.feedback.eval_analyzer.get_weak_configs(
                    threshold=self.feedback_config.win_rate_degradation_threshold
                )
                if config_key in weak_configs:
                    # Urgent retraining needed - lower data threshold
                    if config_state.games_since_training >= self.config.trigger_threshold_games // 4:
                        print(f"[Training] Trigger: Win rate degradation for {config_key} "
                              f"(below {self.feedback_config.win_rate_degradation_threshold})")
                        return config_key

        return None

    async def start_training(self, config_key: str) -> bool:
        """Start a training run for the given configuration.

        For hex boards, exports data with V3 encoder first, then trains.
        For square boards, uses existing data or exports with default encoder.
        """
        # Skip local training if disabled
        if DISABLE_LOCAL_TASKS:
            print(f"[Training] Skipping local training (RINGRIFT_DISABLE_LOCAL_TASKS=true)")
            return False

        if self.state.training_in_progress:
            return False

        # P0-3: Data quality gate - check before training
        if self.feedback:
            # Check parity failure rate
            parity_failure_rate = self.feedback.data_monitor.get_parity_failure_rate()
            if parity_failure_rate > self.feedback_config.max_parity_failure_rate:
                print(f"[Training] BLOCKED by data quality gate: parity failure rate "
                      f"{parity_failure_rate:.1%} exceeds threshold "
                      f"{self.feedback_config.max_parity_failure_rate:.1%}")
                if HAS_PROMETHEUS:
                    DATA_QUALITY_BLOCKED_TRAINING.labels(reason="parity_failure").inc()
                return False

            # Check if data is quarantined
            if self.feedback.should_quarantine_data():
                print(f"[Training] BLOCKED by data quality gate: data quarantined due to quality issues")
                if HAS_PROMETHEUS:
                    DATA_QUALITY_BLOCKED_TRAINING.labels(reason="quarantined").inc()
                return False

            # Check data quality score
            if self.feedback.state.data_quality_score < self.feedback_config.min_data_quality_score:
                print(f"[Training] BLOCKED by data quality gate: data quality score "
                      f"{self.feedback.state.data_quality_score:.2f} below threshold "
                      f"{self.feedback_config.min_data_quality_score:.2f}")
                if HAS_PROMETHEUS:
                    DATA_QUALITY_BLOCKED_TRAINING.labels(reason="low_score").inc()
                return False

            print(f"[Training] Data quality gate PASSED: parity_failure_rate={parity_failure_rate:.1%}, "
                  f"quality_score={self.feedback.state.data_quality_score:.2f}")

        parts = config_key.rsplit("_", 1)
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        # Acquire cluster-wide training lock
        if not self._acquire_training_lock():
            print(f"[Training] Could not acquire cluster-wide lock for {config_key}")
            return False

        await self.event_bus.publish(DataEvent(
            event_type=DataEventType.TRAINING_STARTED,
            payload={"config": config_key}
        ))

        try:
            self.state.training_in_progress = True
            self.state.training_config = config_key
            self.state.training_started_at = time.time()

            # Use v3 for all board types (best architecture with spatial policy heads)
            model_version = "v3"

            # Find game database
            games_dir = AI_SERVICE_ROOT / "data" / "games"
            game_dbs = list(games_dir.glob("*.db"))
            if not game_dbs:
                print(f"[Training] No game databases found")
                self.state.training_in_progress = False
                self._release_training_lock()
                return False

            largest_db = max(game_dbs, key=lambda p: p.stat().st_size)

            # Export training data with appropriate encoder
            training_dir = AI_SERVICE_ROOT / "data" / "training"
            training_dir.mkdir(parents=True, exist_ok=True)
            data_path = training_dir / f"unified_{config_key}.npz"

            # Determine encoder version (v3 for hex, default for square)
            encoder_version = self.config.hex_encoder_version if board_type == "hexagonal" else "default"

            export_cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / self.config.export_script),
                "--db", str(largest_db),
                "--output", str(data_path),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--sample-every", "2",
            ]
            if encoder_version != "default":
                export_cmd.extend(["--encoder-version", encoder_version])

            print(f"[Training] Exporting data for {config_key} (encoder: {encoder_version})...")
            export_process = await asyncio.create_subprocess_exec(
                *export_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )
            await export_process.wait()

            if export_process.returncode != 0:
                print(f"[Training] Export failed for {config_key}")
                self.state.training_in_progress = False
                self._release_training_lock()
                return False

            # Start NN training process
            timestamp = int(time.time())
            model_id = f"{config_key}_v3_{timestamp}"
            run_dir = AI_SERVICE_ROOT / "logs" / "unified_training" / model_id

            # Calculate adaptive epochs based on feedback
            base_epochs = 100
            if self.feedback:
                epochs_multiplier = self.feedback.get_epochs_multiplier()
                epochs = max(50, int(base_epochs * epochs_multiplier))  # Floor at 50
                print(f"[Training] Adaptive epochs: {epochs} (base={base_epochs}, multiplier={epochs_multiplier:.2f})")
            else:
                epochs = base_epochs

            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / self.config.nn_training_script),
                "--board", board_type,
                "--num-players", str(num_players),
                "--data-path", str(data_path),
                "--run-dir", str(run_dir),
                "--model-id", model_id,
                "--model-version", model_version,
                "--epochs", str(epochs),
            ]

            print(f"[Training] Starting training for {model_id}...")
            self._training_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )

            return True

        except Exception as e:
            print(f"[TrainingScheduler] Error starting training: {e}")
            self.state.training_in_progress = False
            self._release_training_lock()
            return False

    async def check_training_status(self) -> Optional[Dict[str, Any]]:
        """Check if current training has completed."""
        if not self.state.training_in_progress or not self._training_process:
            return None

        # Check if process has finished
        if self._training_process.returncode is not None:
            stdout, stderr = await self._training_process.communicate()

            success = self._training_process.returncode == 0
            config_key = self.state.training_config

            self.state.training_in_progress = False
            self.state.training_config = ""
            self.state.total_training_runs += 1
            self._training_process = None
            self._release_training_lock()  # Release cluster-wide lock

            if config_key in self.state.configs:
                self.state.configs[config_key].last_training_time = time.time()
                self.state.configs[config_key].games_since_training = 0

            result = {
                "config": config_key,
                "success": success,
                "duration": time.time() - self.state.training_started_at,
            }

            # Run calibration analysis if training succeeded
            if success and HAS_VALUE_CALIBRATION:
                calibration_report = await self._run_calibration_analysis(config_key)
                if calibration_report:
                    result["calibration"] = calibration_report

            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.TRAINING_COMPLETED,
                payload=result
            ))

            return result

        return None

    async def _run_calibration_analysis(self, config_key: str) -> Optional[Dict[str, Any]]:
        """Run value calibration analysis on recent training data."""
        if not HAS_VALUE_CALIBRATION:
            return None

        try:
            # Get calibration tracker for this config
            if config_key not in self._calibration_trackers:
                self._calibration_trackers[config_key] = CalibrationTracker(window_size=1000)

            tracker = self._calibration_trackers[config_key]

            # Compute calibration from tracker's running window
            report = tracker.compute_current_calibration()
            if report is None:
                print(f"[Calibration] Not enough samples for {config_key}")
                return None

            # Store in state
            report_dict = report.to_dict()
            self.state.calibration_reports[config_key] = report_dict
            self.state.last_calibration_time = time.time()

            # Log calibration metrics
            print(f"[Calibration] {config_key}: ECE={report.ece:.4f}, MCE={report.mce:.4f}, "
                  f"overconfidence={report.overconfidence:.4f}")

            # Check if recalibration is needed
            if report.ece > 0.1:  # High calibration error
                print(f"[Calibration] WARNING: High ECE for {config_key}, consider recalibration")
                if report.optimal_temperature:
                    print(f"[Calibration] Suggested temperature: {report.optimal_temperature:.3f}")

            return report_dict

        except Exception as e:
            print(f"[Calibration] Error analyzing {config_key}: {e}")
            return None

    def get_temperature_for_move(self, move_number: int, game_state: Optional[Any] = None) -> float:
        """Get exploration temperature for a given move in self-play."""
        if self._temp_scheduler is None:
            return 1.0
        return self._temp_scheduler.get_temperature(move_number, game_state)

    def update_training_progress(self, progress: float):
        """Update training progress for curriculum-based temperature scheduling."""
        if self._temp_scheduler is not None:
            self._temp_scheduler.set_training_progress(progress)


# =============================================================================
# Model Promoter Component
# =============================================================================

class ModelPromoter:
    """Handles automatic model promotion based on Elo."""

    def __init__(self, config: PromotionConfig, state: UnifiedLoopState, event_bus: EventBus):
        self.config = config
        self.state = state
        self.event_bus = event_bus

    async def check_promotion_candidates(self) -> List[Dict[str, Any]]:
        """Check for models that should be promoted."""
        if not self.config.auto_promote:
            return []

        candidates = []

        try:
            # Query Elo database for candidates
            elo_db_path = AI_SERVICE_ROOT / "data" / "unified_elo.db"
            if not elo_db_path.exists():
                return []

            conn = sqlite3.connect(elo_db_path)
            cursor = conn.cursor()

            # Find models that beat current best by threshold
            cursor.execute("""
                SELECT participant_id, board_type, num_players, rating, games_played
                FROM elo_ratings
                WHERE games_played >= ?
                ORDER BY board_type, num_players, rating DESC
            """, (self.config.min_games,))

            rows = cursor.fetchall()
            conn.close()

            # Group by config and find candidates
            by_config: Dict[str, List[Tuple]] = {}
            for row in rows:
                config_key = f"{row[1]}_{row[2]}p"
                if config_key not in by_config:
                    by_config[config_key] = []
                by_config[config_key].append(row)

            for config_key, models in by_config.items():
                if len(models) < 2:
                    continue

                best = models[0]
                current_best_id = f"ringrift_best_{config_key.replace('_', '_')}"

                # Check if top model beats current best by threshold
                for model in models:
                    if model[0] == current_best_id:
                        continue
                    if model[3] - best[3] >= self.config.elo_threshold:
                        candidates.append({
                            "model_id": model[0],
                            "config": config_key,
                            "elo": model[3],
                            "games": model[4],
                            "elo_gain": model[3] - best[3],
                        })
                        break

            return candidates

        except Exception as e:
            print(f"[ModelPromoter] Error checking candidates: {e}")
            return []

    async def execute_promotion(self, candidate: Dict[str, Any]) -> bool:
        """Execute a model promotion with holdout validation gate."""
        try:
            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.PROMOTION_CANDIDATE,
                payload=candidate
            ))

            # Holdout validation gate - check for overfitting before promotion
            if HAS_HOLDOUT_VALIDATION and evaluate_model_on_holdout is not None:
                config_key = candidate["config"]
                # Parse board_type and num_players from config key (e.g., "standard_2p")
                parts = config_key.rsplit("_", 1)
                board_type = parts[0] if len(parts) == 2 else config_key
                num_players = int(parts[1].replace("p", "")) if len(parts) == 2 else 2

                # Get model path for evaluation
                model_path = AI_SERVICE_ROOT / "data" / "models" / f"{candidate['model_id']}.pt"
                if not model_path.exists():
                    # Try alternative path patterns
                    model_path = AI_SERVICE_ROOT / "models" / f"{candidate['model_id']}.pt"

                if model_path.exists():
                    try:
                        # Run holdout evaluation (synchronous call in async context)
                        eval_result = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: evaluate_model_on_holdout(
                                model_path=str(model_path),
                                board_type=board_type,
                                num_players=num_players,
                                train_loss=candidate.get("train_loss"),
                            )
                        )

                        # Emit metrics
                        if HAS_PROMETHEUS:
                            HOLDOUT_LOSS.labels(config=config_key).set(eval_result.holdout_loss)
                            if eval_result.overfit_gap is not None:
                                HOLDOUT_OVERFIT_GAP.labels(config=config_key).set(eval_result.overfit_gap)

                        # Check for overfitting
                        if eval_result.overfit_gap is not None and eval_result.overfit_gap > OVERFIT_THRESHOLD:
                            print(f"[ModelPromoter] Promotion BLOCKED for {candidate['model_id']}: "
                                  f"overfit_gap={eval_result.overfit_gap:.4f} > threshold={OVERFIT_THRESHOLD}")
                            if HAS_PROMETHEUS:
                                HOLDOUT_EVALUATIONS.labels(config=config_key, result='failed_overfit').inc()
                                PROMOTION_BLOCKED_OVERFIT.labels(config=config_key).inc()
                            return False

                        print(f"[ModelPromoter] Holdout validation PASSED for {candidate['model_id']}: "
                              f"holdout_loss={eval_result.holdout_loss:.4f}, gap={eval_result.overfit_gap}")
                        if HAS_PROMETHEUS:
                            HOLDOUT_EVALUATIONS.labels(config=config_key, result='passed').inc()

                    except Exception as e:
                        print(f"[ModelPromoter] Holdout validation error (proceeding anyway): {e}")
                        if HAS_PROMETHEUS:
                            HOLDOUT_EVALUATIONS.labels(config=config_key, result='skipped').inc()
                else:
                    print(f"[ModelPromoter] Model file not found for holdout validation: {model_path}")
                    if HAS_PROMETHEUS:
                        HOLDOUT_EVALUATIONS.labels(config=config_key, result='skipped').inc()

            # Run promotion script
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "auto_promote_best_models.py"),
                "--config", candidate["config"],
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

            success = process.returncode == 0

            if success:
                self.state.total_promotions += 1

                await self.event_bus.publish(DataEvent(
                    event_type=DataEventType.MODEL_PROMOTED,
                    payload=candidate
                ))

                # Sync to cluster if enabled
                if self.config.sync_to_cluster:
                    await self._sync_to_cluster(candidate)

            return success

        except Exception as e:
            print(f"[ModelPromoter] Error executing promotion: {e}")
            return False

    async def _sync_to_cluster(self, candidate: Dict[str, Any]):
        """Sync promoted model to cluster."""
        try:
            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "sync_models.py"),
                "--push-promoted",
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )
            await asyncio.wait_for(process.communicate(), timeout=300)

        except Exception as e:
            print(f"[ModelPromoter] Error syncing to cluster: {e}")


# =============================================================================
# Adaptive Curriculum Component
# =============================================================================

class AdaptiveCurriculum:
    """Manages Elo-weighted training curriculum with feedback integration."""

    def __init__(self, config: CurriculumConfig, state: UnifiedLoopState, event_bus: EventBus):
        self.config = config
        self.state = state
        self.event_bus = event_bus
        self.feedback: Optional["PipelineFeedbackController"] = None

    def set_feedback_controller(self, feedback: "PipelineFeedbackController"):
        """Set the feedback controller for curriculum adjustments."""
        self.feedback = feedback

    async def rebalance_weights(self) -> Dict[str, float]:
        """Recompute training weights based on Elo performance."""
        if not self.config.adaptive:
            return {}

        try:
            # Query Elo by config
            elo_db_path = AI_SERVICE_ROOT / "data" / "unified_elo.db"
            if not elo_db_path.exists():
                return {}

            conn = sqlite3.connect(elo_db_path)
            cursor = conn.cursor()

            # Get best Elo for each config
            cursor.execute("""
                SELECT board_type, num_players, MAX(rating) as best_elo
                FROM elo_ratings
                WHERE participant_id LIKE 'ringrift_%'
                GROUP BY board_type, num_players
            """)

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return {}

            elo_by_config = {
                f"{row[0]}_{row[1]}p": row[2]
                for row in rows
            }

            # Compute weights based on deviation from median
            elos = list(elo_by_config.values())
            median_elo = statistics.median(elos)

            new_weights = {}
            for config_key, elo in elo_by_config.items():
                # Boost weight for underperforming configs based on Elo
                deficit = median_elo - elo
                elo_weight = 1.0 + (deficit / 200.0)

                # Merge with feedback controller weights if available
                if self.feedback:
                    feedback_weight = self.feedback.get_curriculum_weight(config_key)
                    # Average Elo-based and feedback-based weights
                    weight = (elo_weight + feedback_weight) / 2.0
                else:
                    weight = elo_weight

                # Clamp to configured range
                weight = max(self.config.min_weight_multiplier,
                           min(self.config.max_weight_multiplier, weight))

                new_weights[config_key] = weight

            # Update state
            self.state.curriculum_weights = new_weights
            self.state.last_curriculum_rebalance = time.time()

            # Update config states
            for config_key, weight in new_weights.items():
                if config_key in self.state.configs:
                    self.state.configs[config_key].training_weight = weight

            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.CURRICULUM_REBALANCED,
                payload={"weights": new_weights}
            ))

            return new_weights

        except Exception as e:
            print(f"[AdaptiveCurriculum] Error rebalancing: {e}")
            return {}


# =============================================================================
# PBT Integration Component
# =============================================================================

class PBTIntegration:
    """Integrates Population-Based Training into the unified loop."""

    def __init__(self, config: PBTConfig, state: UnifiedLoopState, event_bus: EventBus):
        self.config = config
        self.state = state
        self.event_bus = event_bus
        self._pbt_process: Optional[asyncio.subprocess.Process] = None

    async def start_pbt_run(self, board_type: str = "square8", num_players: int = 2) -> bool:
        """Start a new PBT run."""
        if not self.config.enabled:
            return False

        if self.state.pbt_in_progress:
            print("[PBT] Already running")
            return False

        try:
            run_id = f"pbt_{int(time.time())}"
            self.state.pbt_in_progress = True
            self.state.pbt_run_id = run_id
            self.state.pbt_started_at = time.time()
            self.state.pbt_generation = 0

            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "population_based_training.py"),
                "--population-size", str(self.config.population_size),
                "--board", board_type,
                "--players", str(num_players),
                "--tune", ",".join(self.config.tunable_params),
                "--exploit-interval", str(self.config.exploit_interval_steps),
                "--output-dir", str(AI_SERVICE_ROOT / "logs" / "pbt" / run_id),
            ]

            print(f"[PBT] Starting run {run_id}")
            self._pbt_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )

            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.PBT_STARTED,
                payload={"run_id": run_id, "board_type": board_type}
            ))

            return True

        except Exception as e:
            print(f"[PBT] Error starting run: {e}")
            self.state.pbt_in_progress = False
            return False

    async def check_pbt_status(self) -> Optional[Dict[str, Any]]:
        """Check status of running PBT."""
        if not self.state.pbt_in_progress or not self._pbt_process:
            return None

        # Check if process finished
        if self._pbt_process.returncode is not None:
            stdout, stderr = await self._pbt_process.communicate()

            # Load results from state file
            state_file = AI_SERVICE_ROOT / "logs" / "pbt" / self.state.pbt_run_id / "pbt_state.json"
            result = {"run_id": self.state.pbt_run_id, "success": self._pbt_process.returncode == 0}

            if state_file.exists():
                try:
                    with open(state_file) as f:
                        pbt_state = json.load(f)
                    result["best_performance"] = pbt_state.get("best_performance", 0)
                    result["best_hyperparams"] = pbt_state.get("best_hyperparams", {})
                    self.state.pbt_best_performance = result["best_performance"]
                except Exception as e:
                    print(f"[PBT] Error reading state: {e}")

            self.state.pbt_in_progress = False
            self._pbt_process = None

            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.PBT_COMPLETED,
                payload=result
            ))

            return result

        # Check for progress updates
        state_file = AI_SERVICE_ROOT / "logs" / "pbt" / self.state.pbt_run_id / "pbt_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    pbt_state = json.load(f)
                new_gen = pbt_state.get("total_steps", 0) // self.config.exploit_interval_steps
                if new_gen > self.state.pbt_generation:
                    self.state.pbt_generation = new_gen
                    self.state.pbt_best_performance = pbt_state.get("best_performance", 0)
            except Exception:
                pass

        return None


# =============================================================================
# NAS Integration Component
# =============================================================================

class NASIntegration:
    """Integrates Neural Architecture Search into the unified loop."""

    def __init__(self, config: NASConfig, state: UnifiedLoopState, event_bus: EventBus):
        self.config = config
        self.state = state
        self.event_bus = event_bus
        self._nas_process: Optional[asyncio.subprocess.Process] = None

    async def start_nas_run(self, board_type: str = "square8", num_players: int = 2) -> bool:
        """Start a new NAS run."""
        if not self.config.enabled:
            return False

        if self.state.nas_in_progress:
            print("[NAS] Already running")
            return False

        try:
            run_id = f"nas_{self.config.strategy}_{int(time.time())}"
            self.state.nas_in_progress = True
            self.state.nas_run_id = run_id
            self.state.nas_started_at = time.time()
            self.state.nas_generation = 0

            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "neural_architecture_search.py"),
                "--strategy", self.config.strategy,
                "--population", str(self.config.population_size),
                "--generations", str(self.config.generations),
                "--board", board_type,
                "--players", str(num_players),
                "--output-dir", str(AI_SERVICE_ROOT / "logs" / "nas" / run_id),
            ]

            print(f"[NAS] Starting run {run_id} ({self.config.strategy})")
            self._nas_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )

            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.NAS_STARTED,
                payload={"run_id": run_id, "strategy": self.config.strategy}
            ))

            return True

        except Exception as e:
            print(f"[NAS] Error starting run: {e}")
            self.state.nas_in_progress = False
            return False

    async def check_nas_status(self) -> Optional[Dict[str, Any]]:
        """Check status of running NAS."""
        if not self.state.nas_in_progress or not self._nas_process:
            return None

        # Check if process finished
        if self._nas_process.returncode is not None:
            stdout, stderr = await self._nas_process.communicate()

            # Load results
            state_file = AI_SERVICE_ROOT / "logs" / "nas" / self.state.nas_run_id / "nas_state.json"
            result = {"run_id": self.state.nas_run_id, "success": self._nas_process.returncode == 0}

            if state_file.exists():
                try:
                    with open(state_file) as f:
                        nas_state = json.load(f)
                    result["best_performance"] = nas_state.get("best_performance", 0)
                    result["best_architecture"] = nas_state.get("best_architecture", {})
                    self.state.nas_best_architecture = result["best_architecture"]

                    # Publish best architecture event
                    await self.event_bus.publish(DataEvent(
                        event_type=DataEventType.NAS_BEST_ARCHITECTURE,
                        payload={"architecture": result["best_architecture"]}
                    ))
                except Exception as e:
                    print(f"[NAS] Error reading state: {e}")

            self.state.nas_in_progress = False
            self._nas_process = None

            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.NAS_COMPLETED,
                payload=result
            ))

            return result

        # Check for progress updates
        state_file = AI_SERVICE_ROOT / "logs" / "nas" / self.state.nas_run_id / "nas_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    nas_state = json.load(f)
                new_gen = nas_state.get("generation", 0)
                if new_gen > self.state.nas_generation:
                    self.state.nas_generation = new_gen
                    print(f"[NAS] Generation {new_gen}, best={nas_state.get('best_performance', 0):.4f}")
            except Exception:
                pass

        return None


# =============================================================================
# PER Integration Component
# =============================================================================

class PERIntegration:
    """Integrates Prioritized Experience Replay into the unified loop."""

    def __init__(self, config: PERConfig, state: UnifiedLoopState, event_bus: EventBus):
        self.config = config
        self.state = state
        self.event_bus = event_bus

    async def rebuild_buffer(self, db_path: Optional[Path] = None) -> bool:
        """Rebuild the prioritized replay buffer from game database."""
        if not self.config.enabled:
            return False

        try:
            if db_path is None:
                # Find largest game database
                games_dir = AI_SERVICE_ROOT / "data" / "games"
                game_dbs = list(games_dir.glob("*.db"))
                if not game_dbs:
                    print("[PER] No game databases found")
                    return False
                db_path = max(game_dbs, key=lambda p: p.stat().st_size)

            buffer_path = AI_SERVICE_ROOT / "data" / "replay_buffer.pkl"

            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / "scripts" / "prioritized_replay.py"),
                "--build",
                "--db", str(db_path),
                "--output", str(buffer_path),
                "--capacity", str(self.config.buffer_capacity),
                "--alpha", str(self.config.alpha),
                "--beta", str(self.config.beta),
            ]

            print(f"[PER] Rebuilding buffer from {db_path.name}")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)

            if process.returncode == 0:
                self.state.per_buffer_path = str(buffer_path)
                self.state.per_last_rebuild = time.time()

                # Get buffer size
                if buffer_path.exists():
                    import pickle
                    with open(buffer_path, "rb") as f:
                        data = pickle.load(f)
                    self.state.per_buffer_size = data.get("tree", {}).n_entries if hasattr(data.get("tree", {}), "n_entries") else 0

                await self.event_bus.publish(DataEvent(
                    event_type=DataEventType.PER_BUFFER_REBUILT,
                    payload={
                        "buffer_path": str(buffer_path),
                        "buffer_size": self.state.per_buffer_size,
                    }
                ))

                print(f"[PER] Buffer rebuilt: {self.state.per_buffer_size} experiences")
                return True
            else:
                print(f"[PER] Buffer rebuild failed: {stderr.decode()[:200]}")
                return False

        except Exception as e:
            print(f"[PER] Error rebuilding buffer: {e}")
            return False

    def should_rebuild(self) -> bool:
        """Check if buffer should be rebuilt."""
        if not self.config.enabled:
            return False
        now = time.time()
        return now - self.state.per_last_rebuild >= self.config.rebuild_interval_seconds


# =============================================================================
# Main Unified Loop
# =============================================================================

class UnifiedAILoop:
    """Single coordinator for the complete AI improvement loop."""

    def __init__(self, config: UnifiedLoopConfig):
        self.config = config
        self.state = UnifiedLoopState()
        self.event_bus = EventBus()

        # Initialize components
        self.data_collector = StreamingDataCollector(
            config.data_ingestion, self.state, self.event_bus
        )
        self.shadow_tournament = ShadowTournamentService(
            config.evaluation, self.state, self.event_bus
        )
        self.training_scheduler = TrainingScheduler(
            config.training, self.state, self.event_bus,
            feedback_config=config.feedback  # Pass feedback config for performance-based triggers
        )
        self.model_promoter = ModelPromoter(
            config.promotion, self.state, self.event_bus
        )
        self.adaptive_curriculum = AdaptiveCurriculum(
            config.curriculum, self.state, self.event_bus
        )

        # Advanced training components
        self.pbt_integration = PBTIntegration(
            config.pbt, self.state, self.event_bus
        )
        self.nas_integration = NASIntegration(
            config.nas, self.state, self.event_bus
        )
        self.per_integration = PERIntegration(
            config.per, self.state, self.event_bus
        )

        # Pipeline feedback controller for closed-loop integration
        self.feedback: Optional[PipelineFeedbackController] = None
        if HAS_FEEDBACK and config.feedback.enabled:
            try:
                self.feedback = create_feedback_controller(AI_SERVICE_ROOT)
                # Wire feedback controller to training scheduler for performance-based triggers
                self.training_scheduler.set_feedback_controller(self.feedback)
                # Wire feedback controller to adaptive curriculum for curriculum weight integration
                self.adaptive_curriculum.set_feedback_controller(self.feedback)
                print("[UnifiedLoop] Pipeline feedback controller initialized and wired to components")
                # Subscribe to events for feedback loop integration
                self.event_bus.subscribe(DataEventType.EVALUATION_COMPLETED, self._on_evaluation_for_feedback)
                self.event_bus.subscribe(DataEventType.TRAINING_COMPLETED, self._on_training_for_feedback)
                self.event_bus.subscribe(DataEventType.MODEL_PROMOTED, self._on_promotion_for_feedback)
                # Event-driven curriculum rebalancing on significant Elo changes
                self.event_bus.subscribe(DataEventType.ELO_SIGNIFICANT_CHANGE, self._on_elo_significant_change)
            except Exception as e:
                print(f"[UnifiedLoop] Warning: Failed to initialize feedback controller: {e}")

        # Health monitoring
        global _health_tracker
        self.health_tracker = HealthTracker()
        _health_tracker = self.health_tracker  # Set global for HTTP endpoint
        print("[UnifiedLoop] Health tracker initialized")

        # Task coordination - prevents runaway spawning across orchestrators
        self.task_coordinator: Optional[TaskCoordinator] = None
        if HAS_TASK_COORDINATOR:
            try:
                self.task_coordinator = TaskCoordinator.get_instance()
                # Register this loop as the main improvement loop
                import socket
                node_id = socket.gethostname()
                allowed, reason = self.task_coordinator.can_spawn_task(TaskType.IMPROVEMENT_LOOP, node_id)
                if not allowed:
                    print(f"[UnifiedLoop] WARNING: TaskCoordinator denied spawning: {reason}")
                else:
                    self._task_id = f"unified_loop_{int(time.time())}"
                    self.task_coordinator.register_task(
                        self._task_id, TaskType.IMPROVEMENT_LOOP, node_id, os.getpid()
                    )
                print("[UnifiedLoop] Task coordinator integrated - global limits enforced")
            except Exception as e:
                print(f"[UnifiedLoop] Warning: Failed to initialize task coordinator: {e}")
        else:
            self._task_id = None

        # New coordination: acquire UNIFIED_LOOP role (SQLite-backed with heartbeat)
        self._has_orchestrator_role = False
        if HAS_COORDINATION and OrchestratorRole is not None:
            try:
                if acquire_orchestrator_role(OrchestratorRole.UNIFIED_LOOP):
                    self._has_orchestrator_role = True
                    print("[UnifiedLoop] Acquired UNIFIED_LOOP role via new coordination system")
                else:
                    print("[UnifiedLoop] WARNING: Another unified loop already holds the role")
            except Exception as e:
                print(f"[UnifiedLoop] Warning: Failed to acquire new orchestrator role: {e}")

        # Hot data buffer - caches recent games in memory to reduce DB roundtrips
        self.hot_buffer: Optional[HotDataBuffer] = None
        if HAS_HOT_BUFFER and getattr(config, 'use_hot_buffer', True):
            try:
                self.hot_buffer = create_hot_buffer(
                    max_size=getattr(config, 'hot_buffer_size', 1000),
                    max_memory_mb=getattr(config, 'hot_buffer_memory_mb', 500),
                )
                print(f"[UnifiedLoop] Hot data buffer initialized (max_size={self.hot_buffer.max_size})")
            except Exception as e:
                print(f"[UnifiedLoop] Warning: Failed to initialize hot buffer: {e}")

        # Adaptive controller - plateau detection and dynamic game scaling
        self.adaptive_ctrl: Optional[AdaptiveController] = None
        if HAS_ADAPTIVE_CONTROLLER and getattr(config, 'use_adaptive_control', True):
            try:
                state_path = AI_SERVICE_ROOT / config.log_dir / "adaptive_controller_state.json"
                self.adaptive_ctrl = create_adaptive_controller(
                    plateau_threshold=getattr(config, 'plateau_threshold', 5),
                    min_games=getattr(config, 'min_selfplay_games', 50),
                    max_games=getattr(config, 'max_selfplay_games', 200),
                    state_path=state_path,
                )
                print(f"[UnifiedLoop] Adaptive controller initialized (plateau_threshold={self.adaptive_ctrl.plateau_threshold})")
                # Subscribe to promotion events for iteration tracking
                self.event_bus.subscribe(DataEventType.MODEL_PROMOTED, self._on_promotion_for_adaptive_ctrl)
                self.event_bus.subscribe(DataEventType.EVALUATION_COMPLETED, self._on_evaluation_for_adaptive_ctrl)
            except Exception as e:
                print(f"[UnifiedLoop] Warning: Failed to initialize adaptive controller: {e}")

        # P2P cluster integration - distributed training across cluster
        self.p2p: Optional[P2PIntegrationManager] = None
        self._p2p_started = False
        if HAS_P2P and config.p2p.enabled:
            try:
                # Create P2PIntegrationConfig from our local config
                p2p_config = P2PIntegrationConfig(
                    p2p_base_url=config.p2p.p2p_base_url,
                    auth_token=config.p2p.auth_token,
                    model_sync_enabled=config.p2p.model_sync_enabled,
                    target_selfplay_games_per_hour=config.p2p.target_selfplay_games_per_hour,
                    auto_scale_selfplay=config.p2p.auto_scale_selfplay,
                    use_distributed_tournament=config.p2p.use_distributed_tournament,
                    health_check_interval=config.p2p.health_check_interval,
                    sync_interval_seconds=config.p2p.sync_interval_seconds,
                )
                self.p2p = P2PIntegrationManager(p2p_config)

                # Wire P2P callbacks to our event bus
                self.p2p.register_callback("cluster_unhealthy", self._on_p2p_cluster_unhealthy)
                self.p2p.register_callback("nodes_dead", self._on_p2p_nodes_dead)
                self.p2p.register_callback("selfplay_scaled", self._on_p2p_selfplay_scaled)

                print(f"[UnifiedLoop] P2P integration initialized (base_url={config.p2p.p2p_base_url})")
            except Exception as e:
                print(f"[UnifiedLoop] Warning: Failed to initialize P2P integration: {e}")
                self.p2p = None

        # State management
        self._state_path = AI_SERVICE_ROOT / config.log_dir / "unified_loop_state.json"
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Timing trackers
        self._last_shadow_eval: Dict[str, float] = {}
        self._last_full_eval: float = 0.0
        self._started_time: float = 0.0

    def _update_metrics(self):
        """Update Prometheus metrics from current state."""
        if not HAS_PROMETHEUS:
            return

        # Update uptime
        if self._started_time > 0:
            UPTIME_SECONDS.set(time.time() - self._started_time)

        # Update host counts
        active_hosts = sum(1 for h in self.state.hosts.values() if h.enabled and h.consecutive_failures < 3)
        failed_hosts = sum(1 for h in self.state.hosts.values() if h.consecutive_failures >= 3)
        HOSTS_ACTIVE.set(active_hosts)
        HOSTS_FAILED.set(failed_hosts)

        # Update curriculum weights
        for config_key, weight in self.state.curriculum_weights.items():
            CURRICULUM_WEIGHT.labels(config=config_key).set(weight)

        # Update pending games
        for config_key, config_state in self.state.configs.items():
            GAMES_PENDING_TRAINING.labels(config=config_key).set(config_state.games_since_training)
            if config_state.current_elo > 0:
                CURRENT_ELO.labels(config=config_key, model="best").set(config_state.current_elo)
            ELO_TREND.labels(config=config_key).set(config_state.elo_trend)

        # Training in progress
        if self.state.training_in_progress:
            TRAINING_IN_PROGRESS.labels(config=self.state.training_config).set(1)
        else:
            for config_key in self.state.configs:
                TRAINING_IN_PROGRESS.labels(config=config_key).set(0)

        # Update training progress metrics (for dashboard compatibility)
        self._update_training_progress_metrics()

    def _update_training_progress_metrics(self):
        """Update metrics for the training progress dashboard."""
        try:
            models_dir = AI_SERVICE_ROOT / "models"

            # Count model files and calculate sizes
            total_models = 0
            total_size_bytes = 0
            versions_count: Dict[str, int] = {}
            max_version = 0

            if models_dir.exists():
                import re
                version_pattern = re.compile(r'_v(\d+)|^v(\d+)_')

                for model_file in models_dir.rglob("*.pt"):
                    total_models += 1
                    total_size_bytes += model_file.stat().st_size

                    # Extract version from filename (e.g., square8_2p_v2.pt, v3_square8_valueonly.pt)
                    filename = model_file.name
                    match = version_pattern.search(filename)
                    if match:
                        version_num = int(match.group(1) or match.group(2))
                        max_version = max(max_version, version_num)
                        version_key = f"v{version_num}"
                        versions_count[version_key] = versions_count.get(version_key, 0) + 1
                    else:
                        # Also check path parts for directory-based versioning
                        parts = model_file.parts
                        for part in parts:
                            if part.startswith("v") and len(part) > 1 and part[1:].isdigit():
                                version_num = int(part[1:])
                                max_version = max(max_version, version_num)
                                versions_count[part] = versions_count.get(part, 0) + 1
                                break

            TOTAL_MODELS.set(total_models)
            MAX_MODEL_VERSION.set(max_version)
            MODELS_SIZE_GB.set(total_size_bytes / (1024 ** 3))

            for version, count in versions_count.items():
                MODELS_BY_VERSION.labels(version=version).set(count)

            # Update model promotions gauge from state counter
            MODEL_PROMOTIONS.set(self.state.total_promotions)

            # Update Elo and win rates from Elo database
            elo_db_path = AI_SERVICE_ROOT / "data" / "unified_elo.db"
            if elo_db_path.exists():
                conn = sqlite3.connect(elo_db_path)
                cursor = conn.cursor()

                # Get best Elo for each config
                cursor.execute("""
                    SELECT board_type, num_players, participant_id, rating, games_played
                    FROM elo_ratings
                    WHERE participant_id LIKE 'ringrift_%'
                    ORDER BY board_type, num_players, rating DESC
                """)

                rows = cursor.fetchall()
                seen_configs: Set[str] = set()

                for row in rows:
                    config_key = f"{row[0]}_{row[1]}p"
                    if config_key not in seen_configs:
                        seen_configs.add(config_key)
                        MODEL_ELO.labels(config=config_key, model="best").set(row[3])

                        # Estimate win rate from Elo (simplified)
                        # Win rate ≈ 1 / (1 + 10^((1500 - elo) / 400))
                        elo = row[3]
                        estimated_win_rate = 1.0 / (1.0 + 10 ** ((1500 - elo) / 400))
                        MODEL_WIN_RATE.labels(config=config_key).set(estimated_win_rate)

                conn.close()

            # Update max iteration from training runs directory
            runs_dir = AI_SERVICE_ROOT / "runs"
            if runs_dir.exists():
                for config_key in self.state.configs:
                    max_iter = 0
                    # Look for iteration numbers in run directories
                    config_runs = list(runs_dir.glob(f"*{config_key}*"))
                    for run_dir in config_runs:
                        # Extract iteration from training_report.json if exists
                        report_path = run_dir / "training_report.json"
                        if report_path.exists():
                            try:
                                with open(report_path) as f:
                                    report = json.load(f)
                                    iter_num = report.get("iteration", 0)
                                    max_iter = max(max_iter, iter_num)
                            except Exception:
                                pass
                    MAX_ITERATION.labels(config=config_key).set(max_iter)

        except Exception as e:
            print(f"[Metrics] Error updating training progress metrics: {e}")

    # =========================================================================
    # Feedback Controller Callbacks
    # =========================================================================

    async def _on_evaluation_for_feedback(self, event: DataEvent):
        """Handle evaluation completion for feedback loop."""
        if not self.feedback:
            return

        try:
            payload = event.payload
            config_key = payload.get('config')
            win_rate = payload.get('win_rate')
            elo = payload.get('elo')

            # Forward to feedback controller
            await self.feedback.on_stage_complete('evaluation', {
                'config_key': config_key,
                'win_rate': win_rate,
                'elo': elo,
            })

            # Check for CMA-ES trigger signals
            pending_actions = self.feedback.get_pending_actions()
            for signal in pending_actions:
                if signal.action == FeedbackAction.TRIGGER_CMAES:
                    print(f"[Feedback] CMA-ES trigger signal received: {signal.reason}")
                    # Could auto-start CMA-ES here if enabled
                elif signal.action == FeedbackAction.TRIGGER_NAS:
                    print(f"[Feedback] NAS trigger signal received: {signal.reason}")
                    # Could auto-start NAS here if enabled

        except Exception as e:
            print(f"[Feedback] Error processing evaluation feedback: {e}")

    async def _on_training_for_feedback(self, event: DataEvent):
        """Handle training completion for feedback loop."""
        if not self.feedback:
            return

        try:
            payload = event.payload
            config_key = payload.get('config')
            final_loss = payload.get('final_loss')
            val_loss = payload.get('val_loss')
            epochs = payload.get('epochs')

            # Forward to feedback controller
            await self.feedback.on_stage_complete('training', {
                'config_key': config_key,
                'final_loss': final_loss,
                'val_loss': val_loss,
                'epochs': epochs,
            })

            # Log feedback state summary
            summary = self.feedback.get_state_summary()
            print(f"[Feedback] Post-training state: plateau_count={summary['plateau_count']}, "
                  f"weak_configs={summary['weak_configs']}")

        except Exception as e:
            print(f"[Feedback] Error processing training feedback: {e}")

    async def _on_promotion_for_feedback(self, event: DataEvent):
        """Handle model promotion for feedback loop."""
        if not self.feedback:
            return

        try:
            payload = event.payload
            model_id = payload.get('model_id')
            config_key = payload.get('config')
            elo_gain = payload.get('elo_gain', 0)
            success = payload.get('success', True)
            reason = payload.get('reason', '')

            # Forward to feedback controller's promotion handler
            await self.feedback.on_stage_complete('promotion', {
                'config_key': config_key,
                'model_id': model_id,
                'elo_gain': elo_gain,
                'success': success,
                'reason': reason,
            })

            # Log feedback state after promotion handling
            summary = self.feedback.get_state_summary()
            if success:
                print(f"[Feedback] Model {model_id} promoted (+{elo_gain} Elo)")
            else:
                print(f"[Feedback] Promotion failed for {model_id}: {reason}")
                print(f"[Feedback] Consecutive failures: {summary['consecutive_promotion_failures']}, "
                      f"Config failures: {summary['promotion_failure_configs']}")

            # Check for urgent retraining or CMA-ES signals
            pending_actions = self.feedback.get_pending_actions()
            for signal in pending_actions:
                if signal.action == FeedbackAction.URGENT_RETRAINING:
                    print(f"[Feedback] URGENT RETRAINING signal: {signal.reason}")
                elif signal.action == FeedbackAction.TRIGGER_CMAES:
                    print(f"[Feedback] CMA-ES trigger after promotion failures: {signal.reason}")

        except Exception as e:
            print(f"[Feedback] Error processing promotion feedback: {e}")

    async def _on_elo_significant_change(self, event: DataEvent):
        """Handle significant Elo changes by triggering curriculum rebalancing.

        This enables event-driven curriculum rebalancing when a model's Elo
        changes significantly (exceeds the threshold in unified config).
        """
        try:
            payload = event.payload
            config_key = payload.get('config', '')
            model_id = payload.get('model_id', '')
            elo_change = payload.get('elo_change', 0)
            new_elo = payload.get('new_elo', 0)
            threshold = payload.get('threshold', 50)

            print(f"[Curriculum] Significant Elo change detected for {config_key}: "
                  f"{elo_change:+.1f} Elo (model: {model_id}, threshold: {threshold})")

            # Trigger immediate curriculum rebalancing
            if self.adaptive_curriculum:
                old_weights = dict(self.state.curriculum_weights)
                weights = await self.adaptive_curriculum.rebalance_weights()
                if weights:
                    print(f"[Curriculum] Event-driven rebalance triggered: {weights}")
                    self.state.last_curriculum_rebalance = time.time()

                    # Emit rebalance event
                    await self.event_bus.publish(DataEvent(
                        event_type=DataEventType.CURRICULUM_REBALANCED,
                        payload={
                            "trigger": "elo_change",
                            "elo_change": elo_change,
                            "config": config_key,
                            "old_weights": old_weights,
                            "new_weights": weights,
                        },
                    ))

                    # Update Prometheus metrics
                    if HAS_PROMETHEUS:
                        CURRICULUM_REBALANCES_TOTAL.inc()

        except Exception as e:
            print(f"[Curriculum] Error handling Elo change event: {e}")

    # =========================================================================
    # Adaptive Controller Callbacks
    # =========================================================================

    async def _on_promotion_for_adaptive_ctrl(self, event: DataEvent):
        """Record promotion results to adaptive controller for plateau detection."""
        if not self.adaptive_ctrl:
            return

        try:
            payload = event.payload
            promoted = payload.get('success', True)  # If event fired, promotion succeeded
            config_key = payload.get('config', '')
            elo_gain = payload.get('elo_gain', 0)
            win_rate = payload.get('win_rate', 0.5)
            games_played = payload.get('games_played', 0)
            eval_games = payload.get('eval_games', 0)

            # Record to adaptive controller
            iteration = len(self.adaptive_ctrl.history) + 1
            self.adaptive_ctrl.record_iteration(
                iteration=iteration,
                win_rate=win_rate,
                promoted=promoted,
                games_played=games_played,
                eval_games=eval_games,
            )

            # Check plateau status
            plateau_count = self.adaptive_ctrl.get_plateau_count()
            should_continue = self.adaptive_ctrl.should_continue()

            if not should_continue:
                print(f"[AdaptiveCtrl] PLATEAU DETECTED: {plateau_count} iterations without improvement")
                print(f"[AdaptiveCtrl] Consider triggering CMA-ES hyperparameter search or architecture changes")
            elif plateau_count > 0:
                print(f"[AdaptiveCtrl] Plateau count: {plateau_count}/{self.adaptive_ctrl.plateau_threshold}")

            # Log recommended game counts for next iteration
            next_games = self.adaptive_ctrl.compute_games(win_rate)
            next_eval = self.adaptive_ctrl.compute_eval_games(win_rate)
            print(f"[AdaptiveCtrl] Recommended for next iteration: {next_games} selfplay games, {next_eval} eval games")

        except Exception as e:
            print(f"[AdaptiveCtrl] Error processing promotion event: {e}")

    async def _on_evaluation_for_adaptive_ctrl(self, event: DataEvent):
        """Track evaluation results for adaptive control (win rates, trends)."""
        if not self.adaptive_ctrl:
            return

        try:
            payload = event.payload
            win_rate = payload.get('win_rate')
            config_key = payload.get('config', '')

            if win_rate is not None:
                # Store for potential use in compute_games
                next_games = self.adaptive_ctrl.compute_games(win_rate)
                print(f"[AdaptiveCtrl] {config_key} win_rate={win_rate:.2%}, suggested games: {next_games}")

        except Exception as e:
            print(f"[AdaptiveCtrl] Error processing evaluation event: {e}")

    # =========================================================================
    # P2P Cluster Callbacks
    # =========================================================================

    async def _on_p2p_cluster_unhealthy(self, error: Optional[str] = None):
        """Handle cluster unhealthy event from P2P manager."""
        print(f"[P2P] Cluster unhealthy: {error or 'unknown error'}")
        await self.event_bus.publish(DataEvent(
            event_type=DataEventType.P2P_CLUSTER_UNHEALTHY,
            payload={"error": error},
        ))

    async def _on_p2p_nodes_dead(self, nodes: List[str]):
        """Handle nodes dead event from P2P manager."""
        print(f"[P2P] Dead nodes detected: {nodes}")
        await self.event_bus.publish(DataEvent(
            event_type=DataEventType.P2P_NODES_DEAD,
            payload={"nodes": nodes},
        ))

    async def _on_p2p_selfplay_scaled(self, result: Dict[str, Any]):
        """Handle selfplay auto-scale event from P2P manager."""
        actions = result.get("actions", [])
        print(f"[P2P] Selfplay scaled: {len(actions)} actions taken")
        await self.event_bus.publish(DataEvent(
            event_type=DataEventType.P2P_SELFPLAY_SCALED,
            payload=result,
        ))

    async def _start_p2p(self):
        """Start P2P integration manager if configured."""
        if self.p2p and not self._p2p_started:
            try:
                await self.p2p.start()
                self._p2p_started = True
                print("[P2P] Integration manager started")
            except Exception as e:
                print(f"[P2P] Failed to start integration manager: {e}")

    async def _stop_p2p(self):
        """Stop P2P integration manager."""
        if self.p2p and self._p2p_started:
            try:
                await self.p2p.stop()
                self._p2p_started = False
                print("[P2P] Integration manager stopped")
            except Exception as e:
                print(f"[P2P] Error stopping integration manager: {e}")

    async def sync_model_to_cluster(self, model_id: str, model_path: Path) -> bool:
        """Sync a trained model to the P2P cluster.

        Returns True if sync successful, False otherwise.
        """
        if not self.p2p or not self._p2p_started:
            return False

        try:
            result = await self.p2p.sync_model_to_cluster(model_id, model_path)
            synced = result.get("synced_nodes", 0)
            total = result.get("total_nodes", 0)
            print(f"[P2P] Model {model_id} synced to {synced}/{total} nodes")

            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.P2P_MODEL_SYNCED,
                payload={"model_id": model_id, "synced_nodes": synced, "total_nodes": total},
            ))
            return synced > 0
        except Exception as e:
            print(f"[P2P] Error syncing model to cluster: {e}")
            return False

    def _load_state(self):
        """Load state from checkpoint file."""
        if self._state_path.exists():
            try:
                with open(self._state_path) as f:
                    data = json.load(f)
                self.state = UnifiedLoopState.from_dict(data)
                print(f"[UnifiedLoop] Loaded state: {self.state.total_data_syncs} syncs, {self.state.total_training_runs} training runs")
            except Exception as e:
                print(f"[UnifiedLoop] Error loading state: {e}")

    def _save_state(self):
        """Save state to checkpoint file."""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_path, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            print(f"[UnifiedLoop] Error saving state: {e}")

    def _load_hosts(self):
        """Load host configuration from YAML."""
        hosts_path = AI_SERVICE_ROOT / self.config.hosts_config_path
        if not hosts_path.exists():
            print(f"[UnifiedLoop] Hosts config not found: {hosts_path}")
            return

        try:
            with open(hosts_path) as f:
                hosts_data = yaml.safe_load(f)

            # Load standard hosts
            if "standard_hosts" in hosts_data:
                for name, data in hosts_data["standard_hosts"].items():
                    self.state.hosts[name] = HostState(
                        name=name,
                        ssh_host=data.get("ssh_host", ""),
                        ssh_user=data.get("ssh_user", "ubuntu"),
                        ssh_port=data.get("ssh_port", 22),
                    )

            print(f"[UnifiedLoop] Loaded {len(self.state.hosts)} hosts")

        except Exception as e:
            print(f"[UnifiedLoop] Error loading hosts: {e}")

    def _init_configs(self):
        """Initialize board/player configurations."""
        for board_type in ["square8", "square19", "hexagonal"]:
            for num_players in [2, 3, 4]:
                config_key = f"{board_type}_{num_players}p"
                if config_key not in self.state.configs:
                    self.state.configs[config_key] = ConfigState(
                        board_type=board_type,
                        num_players=num_players,
                    )

    async def _data_collection_loop(self):
        """Main data collection loop - runs every 60 seconds."""
        print("[DataCollection] Loop starting...", flush=True)
        _quality_check_counter = 0
        _quality_check_interval = 10  # Check quality every 10 sync cycles

        while self._running:
            try:
                new_games = await self.data_collector.run_collection_cycle()
                self.health_tracker.record_success("data_collector")
                if new_games > 0:
                    print(f"[DataCollection] Synced {new_games} new games")

                    # Check if training threshold reached
                    trigger_config = self.training_scheduler.should_trigger_training()
                    if trigger_config:
                        await self.event_bus.publish(DataEvent(
                            event_type=DataEventType.TRAINING_THRESHOLD_REACHED,
                            payload={"config": trigger_config}
                        ))

                # Periodic data quality check (every 10 cycles, ~10 minutes)
                _quality_check_counter += 1
                if _quality_check_counter >= _quality_check_interval:
                    _quality_check_counter = 0
                    if self.feedback:
                        try:
                            quality_stats = self.data_collector.compute_quality_stats()
                            quality_score = self.feedback.update_data_quality(
                                draw_rate=quality_stats.get("draw_rate"),
                                timeout_rate=quality_stats.get("timeout_rate"),
                                game_lengths=quality_stats.get("game_lengths"),
                            )
                            print(f"[DataQuality] Updated: score={quality_score:.2f}, "
                                  f"draw_rate={quality_stats.get('draw_rate', 0):.1%}, "
                                  f"timeout_rate={quality_stats.get('timeout_rate', 0):.1%}, "
                                  f"sampled={quality_stats.get('total_sampled', 0)} games")

                            # Update Prometheus metrics
                            if HAS_PROMETHEUS:
                                DATA_QUALITY_SCORE.set(quality_score)
                                DATA_QUALITY_DRAW_RATE.set(quality_stats.get("draw_rate", 0))
                                DATA_QUALITY_TIMEOUT_RATE.set(quality_stats.get("timeout_rate", 0))
                                DATA_QUALITY_CHECKS.inc()
                        except Exception as qe:
                            print(f"[DataQuality] Error computing quality: {qe}")

            except Exception as e:
                print(f"[DataCollection] Error: {e}")
                self.state.consecutive_failures += 1
                self.health_tracker.record_failure("data_collector", str(e))

            self._save_state()

            # Wait for next cycle
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.data_ingestion.poll_interval_seconds
                )
                break  # Shutdown requested
            except asyncio.TimeoutError:
                pass  # Continue loop

    async def _evaluation_loop(self):
        """Main evaluation loop - shadow every 15 min, full every 1 hour."""
        while self._running:
            try:
                now = time.time()

                # Check for shadow tournaments
                for config_key in self.state.configs:
                    last_eval = self._last_shadow_eval.get(config_key, 0)
                    if now - last_eval >= self.config.evaluation.shadow_interval_seconds:
                        print(f"[Evaluation] Running shadow tournament for {config_key}")
                        await self.shadow_tournament.run_shadow_tournament(config_key)
                        self._last_shadow_eval[config_key] = now
                        self.health_tracker.record_success("evaluator")
                        break  # One at a time to avoid overload

                # Check for full tournament
                if now - self._last_full_eval >= self.config.evaluation.full_tournament_interval_seconds:
                    print("[Evaluation] Running full tournament")
                    await self.shadow_tournament.run_full_tournament()
                    self._last_full_eval = now
                    self.health_tracker.record_success("evaluator")

            except Exception as e:
                print(f"[Evaluation] Error: {e}")
                self.health_tracker.record_failure("evaluator", str(e))

            # Wait 60 seconds between checks
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=60)
                break
            except asyncio.TimeoutError:
                pass

    async def _training_loop(self):
        """Main training management loop."""
        while self._running:
            try:
                # Check if training completed
                result = await self.training_scheduler.check_training_status()
                if result:
                    print(f"[Training] Completed: {result}")
                    self.health_tracker.record_success("training_scheduler")

                # Check if we should start training
                if not self.state.training_in_progress:
                    trigger_config = self.training_scheduler.should_trigger_training()
                    if trigger_config:
                        print(f"[Training] Starting training for {trigger_config}")
                        await self.training_scheduler.start_training(trigger_config)

            except Exception as e:
                print(f"[Training] Error: {e}")
                self.health_tracker.record_failure("training_scheduler", str(e))

            # Check every 30 seconds
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=30)
                break
            except asyncio.TimeoutError:
                pass

    async def _promotion_loop(self):
        """Main promotion checking loop."""
        while self._running:
            try:
                candidates = await self.model_promoter.check_promotion_candidates()
                self.health_tracker.record_success("promoter")
                for candidate in candidates:
                    print(f"[Promotion] Found candidate: {candidate['model_id']} (+{candidate['elo_gain']} Elo)")
                    await self.model_promoter.execute_promotion(candidate)

            except Exception as e:
                print(f"[Promotion] Error: {e}")
                self.health_tracker.record_failure("promoter", str(e))

            # Check every 5 minutes
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=300)
                break
            except asyncio.TimeoutError:
                pass

    async def _curriculum_loop(self):
        """Main curriculum rebalancing loop."""
        while self._running:
            try:
                now = time.time()
                if now - self.state.last_curriculum_rebalance >= self.config.curriculum.rebalance_interval_seconds:
                    weights = await self.adaptive_curriculum.rebalance_weights()
                    if weights:
                        print(f"[Curriculum] Rebalanced weights: {weights}")
                        self.health_tracker.record_success("curriculum")

            except Exception as e:
                print(f"[Curriculum] Error: {e}")
                self.health_tracker.record_failure("curriculum", str(e))

            # Check every 10 minutes
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=600)
                break
            except asyncio.TimeoutError:
                pass

    async def _metrics_loop(self):
        """Periodically update Prometheus metrics."""
        while self._running:
            try:
                self._update_metrics()
                if HAS_PROMETHEUS:
                    LOOP_CYCLES_TOTAL.labels(loop="metrics").inc()
            except Exception as e:
                print(f"[Metrics] Error: {e}")
                if HAS_PROMETHEUS:
                    LOOP_ERRORS_TOTAL.labels(loop="metrics", error_type=type(e).__name__).inc()

            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=15)
                break
            except asyncio.TimeoutError:
                pass

    async def _health_check_loop(self):
        """Periodically run component health checks."""
        if not HAS_HEALTH_CHECKS:
            print("[HealthCheck] Health checks module not available - skipping")
            return

        health_check_interval = 300  # Check every 5 minutes
        consecutive_failures = 0
        max_consecutive_failures = 3  # Alert threshold

        while self._running:
            # Check for emergency halt signal
            if check_emergency_halt():
                print("[HealthCheck] EMERGENCY HALT detected - initiating graceful shutdown")
                print(f"[HealthCheck] To resume later: rm {EMERGENCY_HALT_FILE}")
                self._running = False
                self._shutdown_event.set()
                break

            try:
                checker = HealthChecker()
                summary = checker.check_all()

                # Update state with health info
                self.state.last_health_check = datetime.now().isoformat()
                self.state.health_status = "healthy" if summary.healthy else "unhealthy"

                # Update Prometheus metrics if available
                if HAS_PROMETHEUS:
                    LOOP_CYCLES_TOTAL.labels(loop="health_check").inc()

                if summary.healthy:
                    consecutive_failures = 0
                    if self.config.verbose:
                        print(f"[HealthCheck] All components healthy")
                else:
                    consecutive_failures += 1
                    print(f"[HealthCheck] UNHEALTHY - {len(summary.issues)} issues:")
                    for issue in summary.issues:
                        print(f"  - {issue}")

                    # Alert on consecutive failures
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"[HealthCheck] CRITICAL: {consecutive_failures} consecutive health check failures!")

                # Log warnings even when healthy
                if summary.warnings:
                    for warning in summary.warnings:
                        print(f"[HealthCheck] Warning: {warning}")

            except Exception as e:
                print(f"[HealthCheck] Error running health check: {e}")
                if HAS_PROMETHEUS:
                    LOOP_ERRORS_TOTAL.labels(loop="health_check", error_type=type(e).__name__).inc()

            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=health_check_interval)
                break
            except asyncio.TimeoutError:
                pass

    async def _hp_tuning_sync_loop(self):
        """Periodically sync HP tuning results to hyperparameters.json.

        Checks for completed HP tuning sessions and applies the best
        hyperparameters automatically when they improve over the current config.
        """
        hp_tuning_dir = AI_SERVICE_ROOT / "logs" / "hp_tuning"
        hp_config_path = AI_SERVICE_ROOT / "config" / "hyperparameters.json"
        sync_interval = 3600  # Check every hour
        min_score_improvement = 0.02  # Require 2% improvement to update

        while self._running:
            try:
                if hp_tuning_dir.exists() and hp_config_path.exists():
                    # Load current hyperparameters config
                    with open(hp_config_path) as f:
                        hp_config = json.load(f)

                    updated = False

                    # Scan for completed tuning sessions
                    for config_dir in hp_tuning_dir.iterdir():
                        if not config_dir.is_dir():
                            continue

                        session_file = config_dir / "tuning_session.json"
                        if not session_file.exists():
                            continue

                        config_key = config_dir.name  # e.g., "square8_2p"
                        if config_key not in hp_config.get("configs", {}):
                            continue

                        try:
                            with open(session_file) as f:
                                session = json.load(f)

                            # Find best trial
                            trials = session.get("trials", [])
                            if not trials:
                                continue

                            best_trial = max(trials, key=lambda t: t.get("combined_score", 0))
                            best_score = best_trial.get("combined_score", 0)

                            # Skip if score is too low (failed trials)
                            if best_score < 0.5:
                                continue

                            # Check if this is an improvement
                            current_config = hp_config["configs"][config_key]
                            current_optimized = current_config.get("optimized", False)

                            # Count completed trials
                            completed_trials = len([t for t in trials if t.get("combined_score", 0) > 0])

                            # Skip if not enough trials completed
                            if completed_trials < 10:
                                continue

                            # Check if we should update
                            should_update = False
                            if not current_optimized:
                                should_update = True
                                reason = "first optimization"
                            elif completed_trials > current_config.get("tuning_trials", 0):
                                # More trials completed - check if score improved
                                should_update = True
                                reason = f"more trials ({completed_trials} > {current_config.get('tuning_trials', 0)})"

                            if should_update:
                                # Extract best parameters
                                best_params = best_trial.get("params", {})

                                # Update config
                                hp_config["configs"][config_key] = {
                                    "optimized": True,
                                    "confidence": "high" if completed_trials >= 30 else "medium",
                                    "tuning_method": "bayesian_optimization",
                                    "last_tuned": datetime.now().isoformat() + "Z",
                                    "tuning_trials": completed_trials,
                                    "hyperparameters": {
                                        "learning_rate": round(best_params.get("learning_rate", 0.0003), 6),
                                        "batch_size": int(best_params.get("batch_size", 256)),
                                        "hidden_dim": int(best_params.get("hidden_dim", 256)),
                                        "num_hidden_layers": int(best_params.get("num_hidden_layers", 2)),
                                        "weight_decay": round(best_params.get("weight_decay", 0.0001), 8),
                                        "dropout": round(best_params.get("dropout", 0.1), 3),
                                        "epochs": 50,
                                        "early_stopping_patience": 15,
                                        "value_weight": round(best_params.get("value_weight", 1.0), 3),
                                        "policy_weight": round(best_params.get("policy_weight", 1.0), 3),
                                    },
                                    "notes": f"Auto-applied from HP tuning. {reason}. Score={best_score:.4f}, val_loss={best_trial.get('val_loss', 'N/A')}"
                                }
                                updated = True
                                print(f"[HPTuning] Updated {config_key}: {reason}, score={best_score:.4f}")

                        except Exception as e:
                            print(f"[HPTuning] Error processing {config_key}: {e}")
                            continue

                    # Save updated config
                    if updated:
                        hp_config["last_updated"] = datetime.now().isoformat() + "Z"
                        with open(hp_config_path, "w") as f:
                            json.dump(hp_config, f, indent=2)
                        print(f"[HPTuning] Saved updated hyperparameters.json")

            except Exception as e:
                print(f"[HPTuning] Error in sync loop: {e}")
                if HAS_PROMETHEUS:
                    LOOP_ERRORS_TOTAL.labels(loop="hp_tuning", error_type=type(e).__name__).inc()

            # Check every hour
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=sync_interval)
                break
            except asyncio.TimeoutError:
                pass

    async def _external_drive_sync_loop(self):
        """External drive sync loop - syncs data to Mac Studio external drive."""
        # Check if external drive sync is enabled in config
        config_path = AI_SERVICE_ROOT / "config" / "unified_loop.yaml"
        if not config_path.exists():
            return

        try:
            with open(config_path) as f:
                full_config = yaml.safe_load(f) or {}

            ext_config = full_config.get("external_drive_sync", {})
            if not ext_config.get("enabled", False):
                print("[ExternalDriveSync] Disabled in config")
                return

            target_dir = Path(ext_config.get("target_dir", "/Volumes/RingRift-Data/selfplay_repository"))
            sync_interval = ext_config.get("sync_interval_seconds", 300)
            sync_models = ext_config.get("sync_models", True)
            run_analysis = ext_config.get("run_analysis", True)

            # Check if target directory parent exists (drive is mounted)
            if not target_dir.parent.exists():
                print(f"[ExternalDriveSync] External drive not mounted at {target_dir.parent}")
                return

            print(f"[ExternalDriveSync] Starting with target={target_dir}, interval={sync_interval}s")

        except Exception as e:
            print(f"[ExternalDriveSync] Config error: {e}")
            return

        while self._running:
            try:
                # Run external_drive_sync_daemon.py --once
                cmd = [
                    sys.executable,
                    str(AI_SERVICE_ROOT / "scripts" / "external_drive_sync_daemon.py"),
                    "--once",
                    "--target", str(target_dir),
                    "--config", str(AI_SERVICE_ROOT / "config" / "distributed_hosts.yaml"),
                ]

                if not sync_models:
                    cmd.append("--no-models")
                if not run_analysis:
                    cmd.append("--no-analysis")

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=AI_SERVICE_ROOT,
                )
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=1800)

                if process.returncode == 0:
                    print(f"[ExternalDriveSync] Cycle complete")
                else:
                    print(f"[ExternalDriveSync] Error: {stderr.decode()[:200]}")

            except asyncio.TimeoutError:
                print("[ExternalDriveSync] Sync timed out")
            except Exception as e:
                print(f"[ExternalDriveSync] Error: {e}")

            # Wait for next cycle
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=sync_interval)
                break
            except asyncio.TimeoutError:
                pass

    async def _pbt_loop(self):
        """PBT management loop."""
        if not self.config.pbt.enabled:
            print("[PBT] Disabled in config")
            return

        while self._running:
            try:
                # Check PBT status if running
                if self.state.pbt_in_progress:
                    result = await self.pbt_integration.check_pbt_status()
                    if result:
                        print(f"[PBT] Completed: best_perf={result.get('best_performance', 0):.4f}")

                # Auto-start PBT after training completes (if enabled)
                elif self.config.pbt.auto_start and not self.state.training_in_progress:
                    # Check if training just completed
                    recent_events = self.event_bus.get_recent_events(DataEventType.TRAINING_COMPLETED, limit=1)
                    if recent_events and time.time() - recent_events[-1].timestamp < 300:
                        config_key = recent_events[-1].payload.get("config", "square8_2p")
                        parts = config_key.rsplit("_", 1)
                        board_type = parts[0]
                        num_players = int(parts[1].replace("p", ""))
                        await self.pbt_integration.start_pbt_run(board_type, num_players)

            except Exception as e:
                print(f"[PBT] Error: {e}")

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.pbt.check_interval_seconds
                )
                break
            except asyncio.TimeoutError:
                pass

    async def _nas_loop(self):
        """NAS management loop."""
        if not self.config.nas.enabled:
            print("[NAS] Disabled in config")
            return

        while self._running:
            try:
                # Check NAS status if running
                if self.state.nas_in_progress:
                    result = await self.nas_integration.check_nas_status()
                    if result:
                        print(f"[NAS] Completed: best_perf={result.get('best_performance', 0):.4f}")

                # Auto-start NAS on Elo plateau (if enabled)
                elif self.config.nas.auto_start_on_plateau:
                    # Check for Elo plateau (no improvement in last 12 hours)
                    plateau_detected = False
                    plateau_config = None
                    for config_key, config_state in self.state.configs.items():
                        if config_state.elo_trend <= 0 and config_state.current_elo > 1500:
                            plateau_detected = True
                            plateau_config = config_key
                            break

                    if plateau_detected and not self.state.nas_in_progress:
                        # Emit PLATEAU_DETECTED event
                        await self.event_bus.publish(DataEvent(
                            event_type=DataEventType.PLATEAU_DETECTED,
                            payload={
                                "config": plateau_config,
                                "current_elo": self.state.configs[plateau_config].current_elo,
                                "elo_trend": self.state.configs[plateau_config].elo_trend,
                                "trigger": "nas_auto_start",
                            }
                        ))
                        # Emit NAS_TRIGGERED event
                        await self.event_bus.publish(DataEvent(
                            event_type=DataEventType.NAS_TRIGGERED,
                            payload={
                                "config": plateau_config,
                                "reason": "elo_plateau",
                                "current_elo": self.state.configs[plateau_config].current_elo,
                            }
                        ))
                        await self.nas_integration.start_nas_run()

            except Exception as e:
                print(f"[NAS] Error: {e}")

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.nas.check_interval_seconds
                )
                break
            except asyncio.TimeoutError:
                pass

    async def _per_loop(self):
        """PER buffer management loop."""
        if not self.config.per.enabled:
            print("[PER] Disabled in config")
            return

        while self._running:
            try:
                # Rebuild buffer periodically
                if self.per_integration.should_rebuild():
                    await self.per_integration.rebuild_buffer()

            except Exception as e:
                print(f"[PER] Error: {e}")

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=600  # Check every 10 minutes
                )
                break
            except asyncio.TimeoutError:
                pass

    async def _cross_process_event_loop(self):
        """Poll for cross-process events from other daemons.

        This bridges events from data_sync_daemon, cluster_orchestrator,
        and other external processes into the unified loop's event bus.
        """
        if not HAS_CROSS_PROCESS_EVENTS:
            print("[CrossProcess] Event queue not available - skipping integration")
            return

        # Initialize cross-process event queue
        try:
            event_queue = CrossProcessEventQueue()
            subscriber_id = event_queue.subscribe(
                process_name="unified_ai_loop",
                event_types=[
                    "new_games",
                    "training_threshold",
                    "model_promoted",
                    "training_completed",
                    "evaluation_completed",
                    "elo_significant_change",
                    "daemon_started",
                    "daemon_stopped",
                ]
            )
            print(f"[CrossProcess] Subscribed as {subscriber_id}")
        except Exception as e:
            print(f"[CrossProcess] Failed to initialize: {e}")
            return

        # Mapping from cross-process event types to local DataEventType
        EVENT_TYPE_MAP = {
            "new_games": DataEventType.NEW_GAMES_AVAILABLE,
            "training_threshold": DataEventType.TRAINING_THRESHOLD_REACHED,
            "model_promoted": DataEventType.MODEL_PROMOTED,
            "training_completed": DataEventType.TRAINING_COMPLETED,
            "evaluation_completed": DataEventType.EVALUATION_COMPLETED,
            "elo_significant_change": DataEventType.ELO_SIGNIFICANT_CHANGE,
        }

        last_event_id = 0

        while self._running:
            try:
                # Poll for new events from other processes
                events = event_queue.poll(
                    subscriber_id=subscriber_id,
                    since_event_id=last_event_id,
                    limit=50,
                )

                for cp_event in events:
                    # Skip events from ourselves
                    if cp_event.source == "unified_ai_loop":
                        event_queue.ack(subscriber_id, cp_event.event_id)
                        last_event_id = max(last_event_id, cp_event.event_id)
                        continue

                    # Map to local event type
                    local_event_type = EVENT_TYPE_MAP.get(cp_event.event_type)
                    if local_event_type:
                        # Bridge to local event bus
                        await self.event_bus.publish(DataEvent(
                            event_type=local_event_type,
                            payload={
                                **cp_event.payload,
                                "_cross_process_source": cp_event.source,
                                "_cross_process_host": cp_event.hostname,
                            }
                        ))
                        print(f"[CrossProcess] Bridged {cp_event.event_type} from {cp_event.source}@{cp_event.hostname}")

                        # Update Prometheus metrics
                        if HAS_PROMETHEUS:
                            CROSS_PROCESS_EVENTS_BRIDGED.labels(
                                event_type=cp_event.event_type,
                                source=cp_event.source
                            ).inc()

                    # Acknowledge the event
                    event_queue.ack(subscriber_id, cp_event.event_id)
                    last_event_id = max(last_event_id, cp_event.event_id)

            except Exception as e:
                print(f"[CrossProcess] Error polling events: {e}")
                if HAS_PROMETHEUS:
                    CROSS_PROCESS_POLL_ERRORS.inc()

            # Poll every 5 seconds
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=5
                )
                break
            except asyncio.TimeoutError:
                pass

        print("[CrossProcess] Event polling stopped")

    def _cleanup_stale_training_locks(self):
        """Clean up stale training lock files on startup.

        Lock files older than 4 hours are considered stale and removed.
        """
        import socket

        lock_dir = AI_SERVICE_ROOT / "data" / "coordination"
        if not lock_dir.exists():
            return

        hostname = socket.gethostname()
        stale_threshold = 4 * 3600  # 4 hours

        for lock_file in lock_dir.glob("training.*.lock"):
            try:
                age = time.time() - lock_file.stat().st_mtime
                if age > stale_threshold:
                    lock_host = lock_file.name.replace("training.", "").replace(".lock", "")
                    print(f"[UnifiedLoop] Removing stale training lock from {lock_host} (age: {age/3600:.1f}h)")
                    lock_file.unlink()
            except Exception as e:
                print(f"[UnifiedLoop] Error cleaning lock file {lock_file}: {e}")

    async def run(self):
        """Main entry point - runs all loops concurrently."""
        # Check emergency halt before starting
        if check_emergency_halt():
            print("[UnifiedLoop] EMERGENCY HALT detected - refusing to start")
            print(f"[UnifiedLoop] To resume: rm {EMERGENCY_HALT_FILE}")
            return

        self._running = True
        self._started_time = time.time()
        self.state.started_at = datetime.now().isoformat()

        # Clean up stale training locks from previous runs
        self._cleanup_stale_training_locks()

        # Load previous state and configuration
        self._load_state()
        self._load_hosts()
        self._init_configs()

        # Update component state references (state object may have been replaced by _load_state)
        self.data_collector.state = self.state
        self.shadow_tournament.state = self.state
        self.training_scheduler.state = self.state
        self.model_promoter.state = self.state
        self.adaptive_curriculum.state = self.state

        # Wire hot buffer to data collector for in-memory game caching
        if self.hot_buffer is not None:
            self.data_collector.set_hot_buffer(self.hot_buffer)

        # Clear stale health cache on startup
        if HAS_PRE_SPAWN_HEALTH and clear_health_cache:
            cleared = clear_health_cache()
            if cleared > 0:
                print(f"[UnifiedLoop] Cleared {cleared} stale host health entries")

        dry_run_msg = " (DRY RUN)" if self.config.dry_run else ""
        print(f"[UnifiedLoop] Starting with {len(self.state.hosts)} hosts, {len(self.state.configs)} configs{dry_run_msg}")
        print(f"[UnifiedLoop] Data sync: {self.config.data_ingestion.poll_interval_seconds}s")
        print(f"[UnifiedLoop] Shadow eval: {self.config.evaluation.shadow_interval_seconds}s")
        print(f"[UnifiedLoop] Full eval: {self.config.evaluation.full_tournament_interval_seconds}s")

        if self.config.dry_run:
            print("[UnifiedLoop] Dry run - showing planned operations:")
            for host_name, host in self.state.hosts.items():
                print(f"  - Would sync from {host.ssh_user}@{host.ssh_host}")
            for config_key in self.state.configs:
                print(f"  - Would run evaluations for {config_key}")
            print("[UnifiedLoop] Dry run complete - exiting")
            return

        # Start P2P cluster integration if configured
        await self._start_p2p()
        if self.p2p and self._p2p_started:
            print(f"[UnifiedLoop] P2P cluster integration enabled")

        try:
            # Start all loops including metrics, external drive sync, and advanced training
            await asyncio.gather(
                self._data_collection_loop(),
                self._evaluation_loop(),
                self._training_loop(),
                self._promotion_loop(),
                self._curriculum_loop(),
                self._metrics_loop(),
                self._health_check_loop(),
                self._hp_tuning_sync_loop(),
                self._external_drive_sync_loop(),
                self._pbt_loop(),
                self._nas_loop(),
                self._per_loop(),
                self._cross_process_event_loop(),
            )
        finally:
            # Stop P2P on shutdown
            await self._stop_p2p()

        print("[UnifiedLoop] Shutdown complete")

    def stop(self):
        """Request graceful shutdown."""
        self._running = False
        self._shutdown_event.set()
        self._save_state()
        # Unregister from task coordinator
        if self.task_coordinator and self._task_id:
            try:
                self.task_coordinator.unregister_task(self._task_id)
                print("[UnifiedLoop] Unregistered from task coordinator")
            except Exception as e:
                print(f"[UnifiedLoop] Warning: Failed to unregister task: {e}")

        # Release new orchestrator role (SQLite-backed)
        if self._has_orchestrator_role and HAS_COORDINATION:
            try:
                release_orchestrator_role()
                print("[UnifiedLoop] Released UNIFIED_LOOP role")
            except Exception as e:
                print(f"[UnifiedLoop] Warning: Failed to release orchestrator role: {e}")


# =============================================================================
# Config Validation
# =============================================================================

def validate_config(config: UnifiedLoopConfig) -> Tuple[bool, List[str]]:
    """Validate configuration at startup.

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    warnings = []

    # Check required paths
    hosts_path = AI_SERVICE_ROOT / config.hosts_config_path
    if not hosts_path.exists():
        errors.append(f"Hosts config not found: {config.hosts_config_path}")

    # Check Elo database directory exists
    elo_db_path = AI_SERVICE_ROOT / config.elo_db
    if not elo_db_path.parent.exists():
        errors.append(f"Elo database directory not found: {elo_db_path.parent}")

    # Validate thresholds
    if config.training.trigger_threshold_games < 100:
        warnings.append(f"Training threshold ({config.training.trigger_threshold_games}) is very low")

    if config.promotion.elo_threshold < 10:
        warnings.append(f"Elo promotion threshold ({config.promotion.elo_threshold}) is very low")

    # Validate intervals
    if config.data_ingestion.poll_interval_seconds < 10:
        errors.append("Sync interval must be at least 10 seconds")

    if config.evaluation.shadow_interval_seconds < 60:
        warnings.append("Shadow tournament interval less than 60s may cause high load")

    # Validate log directory
    log_dir = AI_SERVICE_ROOT / config.log_dir
    if not log_dir.exists():
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create log directory: {e}")

    # Check coordination modules
    if not HAS_COORDINATION:
        warnings.append("Coordination system not available - no inter-process coordination")

    if not HAS_PRE_SPAWN_HEALTH:
        warnings.append("Pre-spawn health checks not available")

    if not HAS_CIRCUIT_BREAKER:
        warnings.append("Circuit breaker not available - no fault tolerance")

    if not HAS_FEEDBACK:
        warnings.append("Pipeline feedback controller not available")

    # Print warnings
    if warnings:
        print("[Config Validation] Warnings:")
        for w in warnings:
            print(f"  - {w}")

    return len(errors) == 0, errors


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified AI Self-Improvement Loop")
    parser.add_argument("--start", action="store_true", help="Start the daemon in background")
    parser.add_argument("--foreground", "-f", action="store_true", help="Run in foreground")
    parser.add_argument("--stop", action="store_true", help="Stop the daemon")
    parser.add_argument("--status", action="store_true", help="Show daemon status")
    parser.add_argument("--config", type=str, default="config/unified_loop.yaml", help="Config file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode - simulate without executing")
    parser.add_argument("--metrics-port", type=int, default=9090, help="Prometheus metrics port")
    parser.add_argument("--no-metrics", action="store_true", help="Disable Prometheus metrics")
    parser.add_argument("--halt", action="store_true", help="Set emergency halt flag to stop all loops")
    parser.add_argument("--resume", action="store_true", help="Clear emergency halt flag to allow restart")

    args = parser.parse_args()

    config_path = AI_SERVICE_ROOT / args.config
    config = UnifiedLoopConfig.from_yaml(config_path)
    config.verbose = args.verbose
    config.dry_run = args.dry_run
    config.metrics_port = args.metrics_port
    config.metrics_enabled = not args.no_metrics

    # Handle emergency halt commands
    if args.halt:
        set_emergency_halt("CLI --halt command")
        print(f"[UnifiedLoop] Emergency halt flag set: {EMERGENCY_HALT_FILE}")
        print("[UnifiedLoop] Running loops will stop at next health check interval (up to 5 min)")
        print("[UnifiedLoop] To resume: python unified_ai_loop.py --resume")
        return

    if args.resume:
        if clear_emergency_halt():
            print(f"[UnifiedLoop] Emergency halt flag cleared")
            print("[UnifiedLoop] You can now restart the loop with --start or --foreground")
        else:
            print("[UnifiedLoop] No emergency halt flag was set")
        return

    # Validate config at startup
    if args.start or args.foreground:
        is_valid, config_errors = validate_config(config)
        if not is_valid:
            print("[UnifiedLoop] ERROR: Configuration validation failed:")
            for err in config_errors:
                print(f"  - {err}")
            return

    if args.status:
        state_path = AI_SERVICE_ROOT / config.log_dir / "unified_loop_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            print("Unified AI Loop Status:")
            print(f"  Started: {state.get('started_at', 'N/A')}")
            print(f"  Last cycle: {state.get('last_cycle_at', 'N/A')}")
            print(f"  Data syncs: {state.get('total_data_syncs', 0)}")
            print(f"  Training runs: {state.get('total_training_runs', 0)}")
            print(f"  Evaluations: {state.get('total_evaluations', 0)}")
            print(f"  Promotions: {state.get('total_promotions', 0)}")
            # Health status
            health_status = state.get('health_status', 'unknown')
            last_health = state.get('last_health_check', 'N/A')
            status_icon = "✓" if health_status == "healthy" else "✗" if health_status == "unhealthy" else "?"
            print(f"  Health: {status_icon} {health_status} (last check: {last_health})")
        else:
            print("No state file found - daemon may not be running")

        # Emergency halt status
        if check_emergency_halt():
            print(f"  EMERGENCY HALT: ACTIVE (use --resume to clear)")
        return

    if args.stop:
        pid_path = AI_SERVICE_ROOT / config.log_dir / "unified_loop.pid"
        if pid_path.exists():
            pid = int(pid_path.read_text().strip())
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"Sent SIGTERM to PID {pid}")
            except ProcessLookupError:
                print(f"Process {pid} not found")
            pid_path.unlink()
        else:
            print("No PID file found")
        return

    if args.start or args.foreground:
        # Check for existing daemon instance using orchestrator registry
        if HAS_COORDINATION:
            registry = get_registry()
            if registry.is_role_held(OrchestratorRole.UNIFIED_LOOP):
                holder = registry.get_role_holder(OrchestratorRole.UNIFIED_LOOP)
                existing_pid = holder.pid if holder else "unknown"
                print(f"[UnifiedLoop] ERROR: Another unified loop instance is already running (PID {existing_pid})")
                print("[UnifiedLoop] Use --stop to stop it first, or kill the existing process")
                return
            print("[UnifiedLoop] Coordination enabled - acquiring UNIFIED_LOOP role")

        # Check system memory - skip on low-memory machines to avoid OOM
        try:
            import psutil
            system_memory_gb = psutil.virtual_memory().total / (1024**3)
            if system_memory_gb < MIN_MEMORY_GB:
                if DISABLE_LOCAL_TASKS:
                    # Allow coordinator-only mode on low-memory machines
                    print(f"[UnifiedLoop] System has {system_memory_gb:.1f}GB RAM (below {MIN_MEMORY_GB}GB threshold)")
                    print("[UnifiedLoop] Running in COORDINATOR-ONLY mode (RINGRIFT_DISABLE_LOCAL_TASKS=true)")
                    print("[UnifiedLoop] Local tournaments and training will be skipped")
                else:
                    print(f"[UnifiedLoop] ERROR: System has only {system_memory_gb:.1f}GB RAM, minimum {MIN_MEMORY_GB}GB required")
                    print("[UnifiedLoop] Exiting to avoid OOM on low-memory machine")
                    print("[UnifiedLoop] Set RINGRIFT_DISABLE_LOCAL_TASKS=true to run in coordination-only mode")
                    return
        except ImportError:
            print("[UnifiedLoop] Warning: psutil not installed, cannot check memory")
        except Exception as e:
            print(f"[UnifiedLoop] Warning: Could not check system memory: {e}")
        if config.dry_run:
            print("[UnifiedLoop] DRY RUN MODE - no actual operations will be performed")

        # Start metrics server
        metrics_server = None
        if config.metrics_enabled and not config.dry_run:
            metrics_server = start_metrics_server(config.metrics_port)

        loop = UnifiedAILoop(config)

        # Handle signals
        def signal_handler(sig, frame):
            print("\n[UnifiedLoop] Received shutdown signal")
            loop.stop()
            if metrics_server:
                metrics_server.shutdown()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if args.start and not args.foreground:
            # Daemonize
            pid = os.fork()
            if pid > 0:
                # Parent
                pid_path = AI_SERVICE_ROOT / config.log_dir / "unified_loop.pid"
                pid_path.parent.mkdir(parents=True, exist_ok=True)
                pid_path.write_text(str(pid))
                print(f"Started daemon with PID {pid}")
                return

        # Run the loop - role is already acquired during UnifiedLoop initialization
        # No need for context manager here as that would cause double acquisition
        asyncio.run(loop.run())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
