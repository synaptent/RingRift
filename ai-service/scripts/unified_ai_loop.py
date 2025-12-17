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

# Import refactored configuration and event types
# These were extracted from this file for modularity (Phase 1 refactoring)
from scripts.unified_loop.config import (
    DataIngestionConfig,
    TrainingConfig,
    EvaluationConfig,
    PromotionConfig,
    CurriculumConfig,
    PBTConfig,
    NASConfig,
    PERConfig,
    FeedbackConfig,
    P2PClusterConfig,
    ModelPruningConfig,
    UnifiedLoopConfig,
    DataEventType,
    DataEvent,
    HostState,
    ConfigState,
)

# Import refactored service classes (Phase 2 refactoring)
from scripts.unified_loop.evaluation import ModelPruningService
from scripts.unified_loop.curriculum import AdaptiveCurriculum
from scripts.unified_loop.promotion import ModelPromoter
from scripts.unified_loop.tournament import ShadowTournamentService
from scripts.unified_loop.data_collection import StreamingDataCollector

# Shared database integrity utilities
from app.db.integrity import (
    check_database_integrity,
    check_and_repair_databases,
    recover_corrupted_database,
)

# Unified resource checking utilities (80% max utilization)
try:
    from app.utils.resource_guard import (
        check_disk_space as unified_check_disk,
        check_memory as unified_check_memory,
        check_cpu as unified_check_cpu,
        get_disk_usage,
        get_memory_usage,
        get_cpu_usage,
        can_proceed as resources_can_proceed,
        LIMITS as RESOURCE_LIMITS,
    )
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    unified_check_disk = None
    unified_check_memory = None
    unified_check_cpu = None
    RESOURCE_LIMITS = None

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
        # Resource targets for unified utilization management
        get_resource_targets,
        get_host_targets,
        get_cluster_summary,
        should_scale_up,
        should_scale_down,
        set_backpressure,
    )
    HAS_COORDINATION = True
except ImportError:
    HAS_COORDINATION = False
    OrchestratorRole = None

# Priority job scheduler for critical job prioritization
try:
    from app.coordination import (
        PriorityJobScheduler,
        JobPriority,
        ScheduledJob,
        get_job_scheduler,
        reset_job_scheduler,
        # Curriculum learning
        get_config_game_counts,
        select_curriculum_config,
        get_underserved_configs,
        get_cpu_rich_hosts,
        get_gpu_rich_hosts,
    )
    HAS_JOB_SCHEDULER = True
except ImportError:
    HAS_JOB_SCHEDULER = False
    PriorityJobScheduler = None
    JobPriority = None
    ScheduledJob = None
    get_job_scheduler = None

# Stage event bus for pipeline orchestration
try:
    from app.coordination import (
        StageEventBus,
        StageEvent,
        StageCompletionResult,
        get_stage_event_bus,
        reset_stage_event_bus,
        register_standard_callbacks,
    )
    HAS_STAGE_EVENTS = True
except ImportError:
    HAS_STAGE_EVENTS = False
    StageEventBus = None
    StageEvent = None
    StageCompletionResult = None
    get_stage_event_bus = None

# P2P Backend with leader discovery
try:
    from app.coordination import (
        P2PBackend,
        P2PNodeInfo,
        discover_p2p_leader_url,
        get_p2p_backend,
        HAS_AIOHTTP as P2P_HAS_AIOHTTP,
    )
    HAS_P2P_BACKEND = True
except ImportError:
    HAS_P2P_BACKEND = False
    P2PBackend = None
    discover_p2p_leader_url = None
    get_p2p_backend = None

# Promotion controller for unified promotion decisions
try:
    from app.training import (
        PromotionController,
        PromotionType,
        PromotionCriteria,
        PromotionDecision,
        get_promotion_controller,
    )
    HAS_PROMOTION_CONTROLLER = True
except ImportError:
    HAS_PROMOTION_CONTROLLER = False
    PromotionController = None
    PromotionType = None
    get_promotion_controller = None

# Resource optimizer for cooperative cluster-wide utilization targeting
try:
    from app.coordination.resource_optimizer import (
        ResourceOptimizer,
        NodeResources,
        ClusterState,
        OptimizationResult,
        ScaleAction,
        get_resource_optimizer,
        get_optimal_concurrency,
        get_cluster_utilization as get_optimizer_cluster_utilization,
        # Rate negotiation for cooperative utilization (60-80% target)
        negotiate_selfplay_rate,
        get_current_selfplay_rate,
        apply_feedback_adjustment,
        get_utilization_status,
        # Data-aware config weighting
        update_config_weights,
        get_config_weights,
    )
    HAS_RESOURCE_OPTIMIZER = True
except ImportError:
    HAS_RESOURCE_OPTIMIZER = False
    ResourceOptimizer = None
    NodeResources = None
    ClusterState = None
    OptimizationResult = None
    ScaleAction = None
    get_resource_optimizer = None
    get_optimal_concurrency = None
    negotiate_selfplay_rate = None
    get_current_selfplay_rate = None
    apply_feedback_adjustment = None
    get_utilization_status = None
    update_config_weights = None
    get_config_weights = None

# Feedback accelerator for positive feedback maximization
try:
    from app.training.feedback_accelerator import (
        FeedbackAccelerator,
        MomentumState,
        TrainingIntensity,
        TrainingDecision,
        get_feedback_accelerator,
        should_trigger_training as accelerator_should_trigger,
        get_training_intensity,
        record_elo_update,
        record_games_generated,
        record_training_complete,
        record_promotion,
        get_curriculum_weights as get_momentum_curriculum_weights,
        get_selfplay_rate_recommendation,
        get_aggregate_selfplay_recommendation,
    )
    HAS_FEEDBACK_ACCELERATOR = True
except ImportError:
    HAS_FEEDBACK_ACCELERATOR = False
    FeedbackAccelerator = None
    MomentumState = None
    TrainingIntensity = None
    get_feedback_accelerator = None
    accelerator_should_trigger = None
    get_training_intensity = None
    record_elo_update = None
    record_games_generated = None
    record_training_complete = None
    record_promotion = None
    get_momentum_curriculum_weights = None
    get_selfplay_rate_recommendation = None
    get_aggregate_selfplay_recommendation = None

# Improvement optimizer for maximizing AI training throughput
try:
    from app.training.improvement_optimizer import (
        ImprovementOptimizer,
        ImprovementSignal,
        OptimizationRecommendation,
        get_improvement_optimizer,
        should_fast_track_training,
        get_dynamic_threshold as get_optimizer_dynamic_threshold,
        get_evaluation_interval,
        record_promotion_success,
        record_training_complete as record_optimizer_training_complete,
        get_improvement_metrics,
    )
    HAS_IMPROVEMENT_OPTIMIZER = True
except ImportError:
    HAS_IMPROVEMENT_OPTIMIZER = False
    ImprovementOptimizer = None
    ImprovementSignal = None
    OptimizationRecommendation = None
    get_improvement_optimizer = None
    should_fast_track_training = None
    get_optimizer_dynamic_threshold = None
    get_evaluation_interval = None
    record_promotion_success = None
    record_optimizer_training_complete = None
    get_improvement_metrics = None

# Unified execution framework for local and remote commands
try:
    from app.execution.executor import (
        LocalExecutor,
        SSHExecutor,
        ExecutorPool,
        ExecutionResult,
        SSHConfig as ExecutorSSHConfig,
        run_command,
    )
    HAS_EXECUTOR = True
except ImportError:
    HAS_EXECUTOR = False
    LocalExecutor = None
    SSHExecutor = None
    ExecutorPool = None
    ExecutionResult = None
    run_command = None

# Import orchestrator backends for unified execution strategy
try:
    from app.execution.backends import (
        OrchestratorBackend,
        BackendType,
        LocalBackend,
        SSHBackend,
        WorkerStatus,
        JobResult,
        get_backend,
    )
    HAS_BACKENDS = True
except ImportError:
    HAS_BACKENDS = False
    OrchestratorBackend = None
    BackendType = None
    LocalBackend = None
    SSHBackend = None
    WorkerStatus = None
    JobResult = None
    get_backend = None

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

# Import diverse tournament orchestrator for Elo calibration
try:
    from scripts.run_diverse_tournaments import (
        build_tournament_configs,
        run_tournament_round_local,
        TournamentConfig,
        TournamentResult,
    )
    HAS_DIVERSE_TOURNAMENTS = True
except ImportError:
    HAS_DIVERSE_TOURNAMENTS = False
    build_tournament_configs = None
    run_tournament_round_local = None
    TournamentConfig = None
    TournamentResult = None

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
        FeedbackSignal,
        FeedbackSignalRouter,
        create_feedback_controller,
    )
    HAS_FEEDBACK = True
except ImportError:
    HAS_FEEDBACK = False
    PipelineFeedbackController = None
    FeedbackAction = None
    FeedbackSignal = None
    FeedbackSignalRouter = None

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
        gate_on_cluster_health,
        is_cluster_healthy,
        HealthStatus as PreSpawnHealthStatus,
    )
    HAS_PRE_SPAWN_HEALTH = True
except ImportError:
    HAS_PRE_SPAWN_HEALTH = False
    is_host_healthy = None
    check_host_health = None
    get_healthy_hosts = None
    gate_on_cluster_health = None
    is_cluster_healthy = None
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
        set_host_breaker_callback,
    )
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False
    CircuitBreaker = None
    CircuitState = None
    CircuitOpenError = None
    get_host_breaker = None
    get_training_breaker = None
    set_host_breaker_callback = None

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

# =============================================================================
# Consolidated Data Sync Modules (unified infrastructure)
# =============================================================================

# Unified Write-Ahead Log for crash-safe data collection
try:
    from app.distributed.unified_wal import (
        UnifiedWAL,
        WALEntry,
        WALEntryType,
        WALEntryStatus,
        WALStats,
        get_unified_wal,
    )
    HAS_UNIFIED_WAL = True
except ImportError:
    HAS_UNIFIED_WAL = False
    UnifiedWAL = None
    WALEntry = None
    WALEntryType = None
    get_unified_wal = None

# Unified Manifest for game deduplication and host state tracking
try:
    from app.distributed.unified_manifest import (
        DataManifest,
        HostSyncState,
        SyncHistoryEntry,
        DeadLetterEntry,
        ManifestStats,
        create_manifest,
    )
    HAS_UNIFIED_MANIFEST = True
except ImportError:
    HAS_UNIFIED_MANIFEST = False
    DataManifest = None
    HostSyncState = None
    create_manifest = None

# Host classification for storage type detection and sync profiles
try:
    from app.distributed.host_classification import (
        StorageType,
        HostTier,
        HostSyncProfile,
        classify_host_storage,
        classify_host_tier,
        get_ephemeral_hosts,
        create_sync_profile,
        create_sync_profiles,
    )
    HAS_HOST_CLASSIFICATION = True
except ImportError:
    HAS_HOST_CLASSIFICATION = False
    StorageType = None
    HostTier = None
    HostSyncProfile = None
    classify_host_storage = None
    get_ephemeral_hosts = None
    create_sync_profile = None

# Gauntlet evaluation and model culling
try:
    from app.tournament.distributed_gauntlet import (
        DistributedNNGauntlet,
        GauntletConfig,
        CONFIG_KEYS as GAUNTLET_CONFIG_KEYS,
        get_gauntlet,
    )
    from app.tournament.model_culling import (
        ModelCullingController,
        get_culling_controller,
    )
    HAS_GAUNTLET = True
except ImportError:
    HAS_GAUNTLET = False
    DistributedNNGauntlet = None
    GauntletConfig = None
    GAUNTLET_CONFIG_KEYS = []
    get_gauntlet = None
    ModelCullingController = None
    get_culling_controller = None

# Elo database synchronization for cluster-wide consistency
try:
    from app.tournament.elo_sync_manager import (
        EloSyncManager,
        get_elo_sync_manager,
        sync_elo_after_games,
        ensure_elo_synced,
    )
    HAS_ELO_SYNC = True
except ImportError:
    HAS_ELO_SYNC = False
    EloSyncManager = None
    get_elo_sync_manager = None
    sync_elo_after_games = None
    ensure_elo_synced = None

# Memory and local task configuration
MIN_MEMORY_GB = 64  # Minimum RAM to run the unified loop
DISABLE_LOCAL_TASKS = os.environ.get("RINGRIFT_DISABLE_LOCAL_TASKS", "").lower() in ("1", "true", "yes", "on")

# Disk capacity limit - stop syncing when disk usage exceeds this percentage
MAX_DISK_USAGE_PERCENT = float(os.environ.get("RINGRIFT_MAX_DISK_PERCENT", "70"))


def get_disk_usage_percent(path: str = None) -> float:
    """Get disk usage percentage for the given path (defaults to AI_SERVICE_ROOT).

    Uses unified resource_guard utilities when available for consistent
    80% max utilization enforcement across the codebase.

    Returns:
        Disk usage as a percentage (0-100), or 100.0 on error.
    """
    if HAS_RESOURCE_GUARD and get_disk_usage is not None:
        try:
            used_pct, _, _ = get_disk_usage(path)
            return used_pct
        except Exception:
            pass
    # Fallback to original implementation
    check_path = path or str(AI_SERVICE_ROOT)
    try:
        stat = os.statvfs(check_path)
        total = stat.f_blocks * stat.f_frsize
        free = stat.f_bavail * stat.f_frsize
        if total <= 0:
            return 100.0
        used = total - free
        return (used / total) * 100.0
    except Exception:
        return 100.0  # Assume full on error to be safe


def check_disk_has_capacity(threshold: float = None) -> Tuple[bool, float]:
    """Check if disk has capacity below threshold for data sync.

    Uses unified resource_guard utilities when available for consistent
    80% max utilization enforcement (70% for disk).

    Args:
        threshold: Max disk usage percentage (defaults to MAX_DISK_USAGE_PERCENT)

    Returns:
        Tuple of (has_capacity: bool, current_percent: float)
    """
    threshold = threshold or MAX_DISK_USAGE_PERCENT
    current = get_disk_usage_percent()
    return (current < threshold, current)


def check_all_resources() -> Tuple[bool, str]:
    """Check if all resources (CPU, memory, disk) are within limits.

    Uses unified 80% max utilization thresholds from resource_guard.

    Returns:
        Tuple of (can_proceed: bool, reason: str)
    """
    if not HAS_RESOURCE_GUARD:
        # Fallback: only check disk
        has_disk, disk_pct = check_disk_has_capacity()
        if not has_disk:
            return False, f"Disk at {disk_pct:.1f}%"
        return True, "OK"

    reasons = []

    # Check disk (70% limit)
    if not unified_check_disk(log_warning=False):
        disk_pct, _, _ = get_disk_usage()
        reasons.append(f"Disk {disk_pct:.1f}%")

    # Check memory (80% limit)
    if not unified_check_memory(log_warning=False):
        mem_pct, _, _ = get_memory_usage()
        reasons.append(f"Memory {mem_pct:.1f}%")

    # Check CPU (80% limit)
    if not unified_check_cpu(log_warning=False):
        cpu_pct, _, _ = get_cpu_usage()
        reasons.append(f"CPU {cpu_pct:.1f}%")

    if reasons:
        return False, ", ".join(reasons)
    return True, "OK"


# =============================================================================
# Database Integrity Checking
# =============================================================================
# Note: check_database_integrity, check_and_repair_databases, and
# recover_corrupted_database are now imported from app.db.integrity
# to avoid code duplication with p2p_orchestrator.py


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

    # Cluster utilization metrics (for 60-80% target tracking)
    CLUSTER_CPU_UTILIZATION = Gauge(
        'ringrift_cluster_cpu_utilization_percent',
        'Average cluster CPU utilization percentage',
        []
    )
    CLUSTER_GPU_UTILIZATION = Gauge(
        'ringrift_cluster_gpu_utilization_percent',
        'Average cluster GPU utilization percentage',
        []
    )
    CLUSTER_MEMORY_UTILIZATION = Gauge(
        'ringrift_cluster_memory_utilization_percent',
        'Average cluster memory utilization percentage',
        []
    )
    CLUSTER_TOTAL_JOBS = Gauge(
        'ringrift_cluster_total_jobs',
        'Total number of jobs running across the cluster',
        []
    )
    CLUSTER_BACKPRESSURE = Gauge(
        'ringrift_cluster_backpressure_factor',
        'Backpressure factor (1.0=none, 0.0=full throttle)',
        []
    )
    HOST_CPU_UTILIZATION = Gauge(
        'ringrift_host_cpu_utilization_percent',
        'Per-host CPU utilization percentage',
        ['host']
    )
    HOST_GPU_UTILIZATION = Gauge(
        'ringrift_host_gpu_utilization_percent',
        'Per-host GPU utilization percentage',
        ['host']
    )
    HOST_JOBS_RUNNING = Gauge(
        'ringrift_host_jobs_running',
        'Number of jobs running on each host',
        ['host']
    )

    # Resource optimizer metrics (60-80% utilization target)
    OPTIMIZER_IN_TARGET = Gauge(
        'ringrift_optimizer_in_target',
        'Cluster utilization in 60-80% target (1=yes, 0=no)',
        ['resource']
    )
    OPTIMIZER_ADJUSTMENT = Gauge(
        'ringrift_optimizer_adjustment',
        'PID controller job adjustment recommendation',
        []
    )
    SELFPLAY_RATE = Gauge(
        'ringrift_selfplay_rate_per_hour',
        'Current negotiated selfplay rate (games per hour)',
        []
    )
    UTILIZATION_STATUS = Gauge(
        'ringrift_utilization_status',
        'Cluster utilization status: -1=below, 0=optimal, 1=above target',
        []
    )
    CONFIG_WEIGHT = Gauge(
        'ringrift_config_weight',
        'Data-aware config weight for selfplay distribution',
        ['config_key']
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


# ============================================
# Types and Configuration (Refactored)
# ============================================
# The following types have been moved to scripts/unified_loop/ for modularity:
# - DataIngestionConfig, TrainingConfig, EvaluationConfig, PromotionConfig,
#   CurriculumConfig, PBTConfig, NASConfig, PERConfig, FeedbackConfig,
#   P2PClusterConfig, ModelPruningConfig, UnifiedLoopConfig (scripts/unified_loop/config.py)
# - DataEventType, DataEvent (scripts/unified_loop/config.py)
# - HostState, ConfigState (scripts/unified_loop/config.py)
#
# They are imported at the top of this file for backward compatibility.
# ============================================


class EventBus:
    """Simple async event bus for component coordination.

    Supports optional cross-process event publishing for events that should
    be visible to other daemons (cluster_orchestrator, data_sync_daemon, etc.)
    """

    # Event types that should be published to cross-process queue
    # These are significant events that other orchestrators/daemons need to know about
    CROSS_PROCESS_EVENT_TYPES = {
        # Success events - coordination across processes
        DataEventType.MODEL_PROMOTED,
        DataEventType.TRAINING_COMPLETED,
        DataEventType.EVALUATION_COMPLETED,
        DataEventType.CURRICULUM_REBALANCED,
        DataEventType.ELO_SIGNIFICANT_CHANGE,
        DataEventType.P2P_MODEL_SYNCED,
        DataEventType.PLATEAU_DETECTED,
        DataEventType.DATA_SYNC_COMPLETED,
        DataEventType.HYPERPARAMETER_UPDATED,
        # Failure events - important for distributed health awareness
        DataEventType.TRAINING_FAILED,
        DataEventType.EVALUATION_FAILED,
        DataEventType.PROMOTION_FAILED,
        DataEventType.DATA_SYNC_FAILED,
        # Host/cluster events - topology awareness
        DataEventType.HOST_ONLINE,
        DataEventType.HOST_OFFLINE,
        DataEventType.DAEMON_STARTED,
        DataEventType.DAEMON_STOPPED,
        # Trigger events - distributed optimization
        DataEventType.CMAES_TRIGGERED,
        DataEventType.NAS_TRIGGERED,
        DataEventType.TRAINING_THRESHOLD_REACHED,
    }

    def __init__(self, cross_process_queue: Optional["CrossProcessEventQueue"] = None):
        self._subscribers: Dict[DataEventType, List[Callable]] = {}
        self._event_history: List[DataEvent] = []
        self._max_history = 1000
        self._cross_process_queue = cross_process_queue

    def set_cross_process_queue(self, queue: "CrossProcessEventQueue"):
        """Set the cross-process queue for outbound event publishing."""
        self._cross_process_queue = queue

    def subscribe(self, event_type: DataEventType, callback: Callable):
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    async def publish(self, event: DataEvent):
        """Publish an event to all subscribers and optionally to cross-process queue."""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        # Publish to cross-process queue for significant events
        # Skip if event was already bridged from cross-process (avoid echo)
        if (
            self._cross_process_queue is not None
            and event.event_type in self.CROSS_PROCESS_EVENT_TYPES
            and "_cross_process_source" not in event.payload
        ):
            try:
                self._cross_process_queue.publish(
                    event_type=event.event_type.value,
                    payload=event.payload,
                    source="unified_ai_loop"
                )
            except Exception as e:
                print(f"[EventBus] Failed to publish to cross-process queue: {e}")

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


class ConfigPriorityQueue:
    """Priority queue for config evaluation and training based on performance.

    Prioritizes configs that need the most attention:
    1. Negative Elo trend (getting worse) - highest priority
    2. Haven't been evaluated recently
    3. Lower Elo (need more improvement)
    4. Higher games_since_training (more data available)
    5. Fewer trained models (configs with no/few models need training first)

    This ensures resources are focused on configs that will benefit most
    from evaluation and training, maximizing the feedback loop efficiency.
    """

    def __init__(self):
        self._priority_cache: Dict[str, float] = {}
        self._last_recalc: float = 0.0
        self._recalc_interval: float = 60.0  # Recalculate every 60 seconds
        self._trained_model_counts: Dict[str, int] = {}
        self._model_counts_last_update: float = 0.0

    def _update_trained_model_counts(self) -> None:
        """Fetch count of trained models per config from filesystem and Elo database.

        Uses filesystem as primary source for accuracy, supplemented by Elo DB.
        This ensures underrepresented configs get proper priority.
        """
        now = time.time()
        # Update every 2 minutes (more frequent for accurate prioritization)
        if now - self._model_counts_last_update < 120:
            return

        counts: Dict[str, int] = {}

        # Initialize all configs to 0
        for board in ["square8", "square19", "hexagonal"]:
            for players in [2, 3, 4]:
                counts[f"{board}_{players}p"] = 0

        try:
            # Primary: Count models from filesystem (most accurate)
            models_dir = AI_SERVICE_ROOT / "models"
            if models_dir.exists():
                for model_file in models_dir.glob("*.pt"):
                    name = model_file.stem.lower()
                    # Match patterns like square8_2p, hex_3p, square19_4p
                    for board in ["square8", "square19", "hex"]:
                        for players in [2, 3, 4]:
                            patterns = [
                                f"{board}_{players}p",
                                f"{board}{players}p",
                                f"{'hexagonal' if board == 'hex' else board}_{players}p",
                            ]
                            if any(p in name for p in patterns):
                                config_key = f"{'hexagonal' if board == 'hex' else board}_{players}p"
                                counts[config_key] = counts.get(config_key, 0) + 1
                                break

                # Also count .pth files
                for model_file in models_dir.glob("*.pth"):
                    name = model_file.stem.lower()
                    for board in ["square8", "square19", "hex"]:
                        for players in [2, 3, 4]:
                            patterns = [
                                f"{board}_{players}p",
                                f"{board}{players}p",
                                f"{'hexagonal' if board == 'hex' else board}_{players}p",
                            ]
                            if any(p in name for p in patterns):
                                config_key = f"{'hexagonal' if board == 'hex' else board}_{players}p"
                                counts[config_key] = counts.get(config_key, 0) + 1
                                break

            # Supplement with Elo database counts via centralized service
            if get_elo_service is not None:
                try:
                    elo_svc = get_elo_service()
                    rows = elo_svc.execute_query("""
                        SELECT board_type, num_players, COUNT(DISTINCT participant_id) as model_count
                        FROM elo_ratings
                        WHERE (participant_id LIKE 'ringrift_%' OR participant_id LIKE '%_nn_baseline%')
                          AND participant_id NOT LIKE 'baseline_%'
                        GROUP BY board_type, num_players
                    """)
                    for row in rows:
                        config_key = f"{row[0]}_{row[1]}p"
                        # Take max of filesystem and DB counts
                        counts[config_key] = max(counts.get(config_key, 0), row[2])
                except Exception as e:
                    print(f"[Priority] Elo service query failed: {e}")

            self._trained_model_counts = counts
            self._model_counts_last_update = now

            # Log model counts for visibility
            if hasattr(self, '_last_model_count_log') and now - self._last_model_count_log < 300:
                pass  # Don't log too frequently
            else:
                self._last_model_count_log = now
                sorted_counts = sorted(counts.items(), key=lambda x: x[1])
                print(f"[Priority] Model counts: {dict(sorted_counts)}")
        except Exception as e:
            print(f"[Priority] Error updating model counts: {e}")
            self._model_counts_last_update = now

    def calculate_priority(
        self,
        config_key: str,
        config_state: ConfigState,
        now: Optional[float] = None
    ) -> float:
        """Calculate priority score for a config (higher = more urgent).

        Priority factors (weights can be tuned):
        - Negative Elo trend: +100 points per -10 Elo trend
        - Time since evaluation: +10 points per 5 minutes
        - Distance from Elo 1800 (target): +50 points per 100 Elo below
        - Games since training: +20 points per 100 games

        Returns:
            Priority score (higher = should be processed first)
        """
        if now is None:
            now = time.time()

        score = 0.0

        # Factor 1: Negative Elo trend (biggest weight - fix regressions fast)
        if config_state.elo_trend < 0:
            score += abs(config_state.elo_trend) * 10  # -10 trend = +100 priority

        # Factor 2: Time since last evaluation
        time_since_eval = now - config_state.last_evaluation_time
        score += (time_since_eval / 300) * 10  # +10 points per 5 minutes

        # Factor 3: Distance from target Elo (1800)
        target_elo = 1800.0
        if config_state.current_elo < target_elo:
            elo_gap = target_elo - config_state.current_elo
            score += (elo_gap / 100) * 50  # +50 points per 100 Elo below target

        # Factor 4: Games since training (more data = ready for training)
        score += (config_state.games_since_training / 100) * 20

        # Factor 5: Bonus for configs that recently had successful promotions
        # (momentum - keep pushing what's working)
        time_since_promotion = now - config_state.last_promotion_time
        if time_since_promotion < 3600:  # Within last hour
            score += 30  # Bonus for recent success

        # Factor 6: MASSIVE bonus for configs with few/no trained models
        # Uses inverse weighting: fewer models = much higher priority
        # This ensures under-represented configs get absolute priority
        self._update_trained_model_counts()
        model_count = self._trained_model_counts.get(config_key, 0)

        # Get max model count across all configs for relative scoring
        max_model_count = max(self._trained_model_counts.values()) if self._trained_model_counts else 1
        max_model_count = max(max_model_count, 1)  # Avoid division by zero

        # Inverse model count scoring: configs with fewer models get MUCH higher priority
        # Formula: base_score * (max_models / (current_models + 1))
        # This creates strong pressure to balance model counts
        if model_count == 0:
            score += 1000  # Highest priority - no trained models at all
        elif model_count == 1:
            score += 800   # Critical - only 1 model
        elif model_count <= 3:
            score += 600   # Very high priority - 2-3 models
        elif model_count <= 6:
            score += 400   # High priority - 4-6 models
        elif model_count <= 10:
            score += 200   # Medium priority - 7-10 models
        else:
            # Low priority for configs with many models
            # Inverse weight: configs with 16 models get ~30 points, with 10 get ~50
            inverse_weight = (max_model_count / (model_count + 1)) * 50
            score += inverse_weight

        # Additional penalty for over-represented configs to create balance pressure
        if max_model_count > 0 and model_count > max_model_count * 0.8:
            score -= 100  # Penalty for configs that already have most models

        return score

    def get_prioritized_configs(
        self,
        configs: Dict[str, ConfigState],
        limit: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """Get configs sorted by priority (highest first).

        Args:
            configs: Dict of config_key -> ConfigState
            limit: Optional limit on number of configs to return

        Returns:
            List of (config_key, priority_score) tuples, sorted by priority
        """
        now = time.time()

        # Recalculate priorities if cache is stale
        if now - self._last_recalc > self._recalc_interval:
            self._priority_cache.clear()
            for config_key, config_state in configs.items():
                self._priority_cache[config_key] = self.calculate_priority(
                    config_key, config_state, now
                )
            self._last_recalc = now

        # Sort by priority (highest first)
        sorted_configs = sorted(
            self._priority_cache.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if limit:
            sorted_configs = sorted_configs[:limit]

        return sorted_configs

    def get_highest_priority_for_training(
        self,
        configs: Dict[str, ConfigState],
        min_games_threshold: int = 100
    ) -> Optional[str]:
        """Get the highest priority config that's ready for training.

        Args:
            configs: Dict of config_key -> ConfigState
            min_games_threshold: Minimum games_since_training to be eligible

        Returns:
            Config key with highest priority that meets threshold, or None
        """
        now = time.time()
        best_key = None
        best_score = -1.0

        for config_key, config_state in configs.items():
            if config_state.games_since_training < min_games_threshold:
                continue

            score = self.calculate_priority(config_key, config_state, now)
            if score > best_score:
                best_score = score
                best_key = config_key

        return best_key

    def invalidate_cache(self, config_key: Optional[str] = None):
        """Invalidate priority cache (call after state changes).

        Args:
            config_key: Specific config to invalidate, or None for all
        """
        if config_key:
            self._priority_cache.pop(config_key, None)
        else:
            self._priority_cache.clear()
            self._last_recalc = 0.0


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

    # Tier gating state (config_key -> current tier)
    # Tiers: D2 -> D4 -> D6 -> D8 (difficulty progression)
    tier_assignments: Dict[str, str] = field(default_factory=dict)
    tier_promotions_count: int = 0
    last_tier_check: float = 0.0

    # Parity validation state
    last_parity_validation: float = 0.0
    parity_validation_passed: bool = True
    parity_games_passed: int = 0
    parity_games_failed: int = 0

    # CMA-ES state
    cmaes_in_progress: bool = False
    cmaes_run_id: str = ""
    cmaes_started_at: float = 0.0
    cmaes_best_weights: Optional[Dict[str, float]] = None

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
            # Tier gating state
            "tier_assignments": self.tier_assignments,
            "tier_promotions_count": self.tier_promotions_count,
            "last_tier_check": self.last_tier_check,
            # Parity validation state
            "last_parity_validation": self.last_parity_validation,
            "parity_validation_passed": self.parity_validation_passed,
            "parity_games_passed": self.parity_games_passed,
            "parity_games_failed": self.parity_games_failed,
            # CMA-ES state
            "cmaes_in_progress": self.cmaes_in_progress,
            "cmaes_run_id": self.cmaes_run_id,
            "cmaes_started_at": self.cmaes_started_at,
            "cmaes_best_weights": self.cmaes_best_weights,
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
                    "calibration_reports", "last_calibration_time", "current_temperature_preset",
                    # Tier gating state
                    "tier_assignments", "tier_promotions_count", "last_tier_check",
                    # Parity validation state
                    "last_parity_validation", "parity_validation_passed",
                    "parity_games_passed", "parity_games_failed",
                    # CMA-ES state
                    "cmaes_in_progress", "cmaes_run_id", "cmaes_started_at", "cmaes_best_weights"]:
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
# EXTRACTED: Now imported from scripts.unified_loop.data_collection (Phase 2 refactoring)
# =============================================================================

# =============================================================================
# Shadow Tournament Component
# EXTRACTED: Now imported from scripts.unified_loop.tournament (Phase 2 refactoring)
# =============================================================================

# =============================================================================
# Model Pruning Service Component
# EXTRACTED: Now imported from scripts.unified_loop.evaluation (Phase 2 refactoring)
# =============================================================================

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
        feedback: Optional["PipelineFeedbackController"] = None,
        config_priority: Optional["ConfigPriorityQueue"] = None
    ):
        self.config = config
        self.state = state
        self.event_bus = event_bus
        self.feedback_config = feedback_config or FeedbackConfig()
        self.feedback = feedback
        self.config_priority = config_priority or ConfigPriorityQueue()
        self._training_process: Optional[asyncio.subprocess.Process] = None
        # Dynamic threshold tracking (for promotion velocity calculation)
        self._promotion_history: List[float] = []  # Timestamps of recent promotions
        self._training_history: List[float] = []   # Timestamps of recent training runs
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



    def _get_dynamic_threshold(self, config_key: str) -> int:
        """Calculate dynamic training threshold based on promotion velocity.

        The threshold adjusts based on:
        1. Promotion velocity - more frequent promotions → lower threshold (faster iteration)
        2. Elo improvement rate - rapid improvement → lower threshold
        3. Time since last promotion - long gap → lower threshold (try to break plateau)

        Returns:
            Adjusted game threshold for training
        """
        base_threshold = self.config.trigger_threshold_games  # Default: 500

        # Track promotion velocity (promotions per hour)
        now = time.time()
        recent_promotions = [t for t in self._promotion_history if now - t < 3600 * 6]  # Last 6 hours
        promotions_per_hour = len(recent_promotions) / 6.0 if recent_promotions else 0

        # Track training velocity
        recent_training = [t for t in self._training_history if now - t < 3600 * 6]
        training_per_hour = len(recent_training) / 6.0 if recent_training else 0

        # Calculate adjustment factor
        adjustment = 1.0

        # Factor 1: Promotion velocity
        # High promotion rate (> 0.5/hour) → we're improving fast, keep threshold low
        # Low promotion rate (< 0.1/hour) → might be stuck, try more frequent training
        if promotions_per_hour > 0.5:
            adjustment *= 0.7  # 30% lower threshold - ride the momentum
        elif promotions_per_hour < 0.1:
            adjustment *= 0.8  # 20% lower threshold - try to break plateau
        else:
            adjustment *= 0.9  # 10% lower - moderate improvement pace

        # Factor 2: Training-to-promotion ratio
        # If we're training a lot but not promoting, increase threshold (save resources)
        if training_per_hour > 2 and promotions_per_hour < 0.2:
            adjustment *= 1.5  # Increase threshold - training not helping
            print(f"[Training] Dynamic: High training ({training_per_hour:.1f}/hr) but low promotion ({promotions_per_hour:.1f}/hr) - increasing threshold")

        # Factor 3: Time since last promotion (staleness)
        config_state = self.state.configs.get(config_key)
        if config_state:
            time_since_promotion = now - config_state.last_promotion_time
            if time_since_promotion > 3600 * 2:  # 2+ hours without promotion
                adjustment *= 0.75  # Try more aggressive training
                print(f"[Training] Dynamic: {time_since_promotion/3600:.1f}h since last promotion for {config_key} - lowering threshold")

        # Apply adjustment with bounds
        dynamic_threshold = int(base_threshold * adjustment)
        min_threshold = base_threshold // 4  # Never go below 25% of base
        max_threshold = base_threshold * 2   # Never go above 200% of base

        final_threshold = max(min_threshold, min(max_threshold, dynamic_threshold))

        # Factor 4: Improvement optimizer positive feedback acceleration
        # When we're on a promotion streak or have high-quality data, push even harder
        if HAS_IMPROVEMENT_OPTIMIZER:
            try:
                optimizer = get_improvement_optimizer()
                optimizer_threshold = optimizer.get_dynamic_threshold(config_key)
                metrics = optimizer.get_improvement_metrics()

                # Take the more aggressive (lower) threshold between local and optimizer
                if optimizer_threshold < final_threshold:
                    streak_info = f"streak={metrics.get('consecutive_promotions', 0)}"
                    print(f"[ImprovementOptimizer] Accelerating threshold for {config_key}: "
                          f"{final_threshold} → {optimizer_threshold} ({streak_info})")
                    final_threshold = optimizer_threshold

                # Additional fast-track check for exceptional performance
                if should_fast_track_training(config_key):
                    fast_threshold = max(min_threshold, final_threshold * 8 // 10)  # 20% faster
                    if fast_threshold < final_threshold:
                        print(f"[ImprovementOptimizer] Fast-tracking {config_key}: {final_threshold} → {fast_threshold}")
                        final_threshold = fast_threshold
            except Exception as e:
                if self.config.verbose:
                    print(f"[ImprovementOptimizer] Error getting threshold: {e}")

        # Factor 5: Cluster utilization-based adjustment for 60-80% target
        # Low utilization → train more aggressively; High utilization → slow down
        if HAS_RESOURCE_OPTIMIZER and get_utilization_status is not None:
            try:
                util_status = get_utilization_status()
                cpu_util = util_status.get('cpu_util', 70)
                gpu_util = util_status.get('gpu_util', 70)
                avg_util = (cpu_util + gpu_util) / 2 if gpu_util > 0 else cpu_util
                if avg_util < 50:
                    final_threshold = max(min_threshold, final_threshold * 6 // 10)
                elif avg_util < 60:
                    final_threshold = max(min_threshold, final_threshold * 8 // 10)
                elif avg_util > 85:
                    final_threshold = min(max_threshold, final_threshold * 12 // 10)
            except Exception:
                pass

        # Factor 6: Underrepresented config priority - CRITICAL for balancing model counts
        # Configs with 0 trained models get drastically lower threshold to bootstrap training
        # This ensures we build models for ALL 9 board/player combinations
        if hasattr(self.config_priority, '_trained_model_counts'):
            self.config_priority._update_trained_model_counts()
            model_count = self.config_priority._trained_model_counts.get(config_key, 0)

            if model_count == 0:
                # NO trained models - minimum viable threshold (50 games)
                # This is critical for bootstrapping underrepresented configs
                bootstrap_threshold = 50
                if final_threshold > bootstrap_threshold:
                    print(f"[Training] BOOTSTRAP: {config_key} has 0 trained models - threshold {final_threshold} → {bootstrap_threshold}")
                    final_threshold = bootstrap_threshold
            elif model_count == 1:
                # Only 1 model - still very underrepresented
                single_model_threshold = min_threshold  # 125 games
                if final_threshold > single_model_threshold:
                    print(f"[Training] UNDERREPRESENTED: {config_key} has 1 model - threshold {final_threshold} → {single_model_threshold}")
                    final_threshold = single_model_threshold
            elif model_count <= 3:
                # 2-3 models - lower threshold to catch up
                catchup_threshold = min_threshold * 3 // 2  # ~188 games
                if final_threshold > catchup_threshold:
                    print(f"[Training] CATCHUP: {config_key} has {model_count} models - threshold {final_threshold} → {catchup_threshold}")
                    final_threshold = catchup_threshold

        if final_threshold != base_threshold:
            print(f"[Training] Dynamic threshold for {config_key}: {final_threshold} (base: {base_threshold}, adj: {adjustment:.2f})")

        return final_threshold

    def record_promotion(self):
        """Record a successful promotion for velocity tracking."""
        now = time.time()
        self._promotion_history.append(now)
        # Keep only last 24 hours
        self._promotion_history = [t for t in self._promotion_history if now - t < 86400]

    def record_training_start(self):
        """Record a training run start for velocity tracking."""
        now = time.time()
        self._training_history.append(now)
        # Keep only last 24 hours
        self._training_history = [t for t in self._training_history if now - t < 86400]

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
        1. Momentum-based acceleration (fastest) - model improving, capitalize on momentum
        2. Game count threshold (traditional) - enough new games collected
        3. Elo plateau detection - model stopped improving
        4. Win rate degradation - model performing worse than threshold
        """
        if self.state.training_in_progress:
            return None

        # Check for cluster-wide training lock (prevent simultaneous training)
        if self.is_training_locked_elsewhere():
            return None

        # Duration-aware scheduling: check if training can be scheduled now
        # Skip peak hours check for dedicated training cluster - always allow training
        # Only check for host availability (concurrent task limits)
        if HAS_COORDINATION:
            import socket
            from app.coordination.duration_scheduler import get_scheduler
            node_id = socket.gethostname()
            scheduler = get_scheduler()
            # Use can_schedule_now with avoid_peak_hours=False for dedicated cluster
            can_schedule, schedule_reason = scheduler.can_schedule_now(
                "training", node_id, avoid_peak_hours=False
            )
            if not can_schedule:
                if self.config.verbose:
                    print(f"[Training] Deferred by duration scheduler: {schedule_reason}")
                return None

        # Health-aware training: defer only when GPU is overloaded (>85% utilization)
        # Training is GPU-bound, so only gate on GPU utilization - not CPU.
        # This allows CPU-bound selfplay and GPU-bound training to run independently,
        # pursuing 70% utilization targets for each resource independently.
        if HAS_RESOURCE_OPTIMIZER and get_utilization_status is not None:
            try:
                util_status = get_utilization_status()
                cpu_util = util_status.get('cpu_util', 70)
                gpu_util = util_status.get('gpu_util', 70)

                # Only defer training if GPU is overloaded - CPU utilization is irrelevant
                # for GPU-bound training tasks
                if gpu_util > 85:
                    if self.state.verbose:
                        print(f"[Training] Deferred due to high GPU utilization "
                              f"(GPU={gpu_util:.1f}%, CPU={cpu_util:.1f}% - CPU ignored for GPU tasks)")
                    return None
            except Exception:
                pass  # Non-critical, proceed with training

        # Cluster health gate: defer training if cluster is degraded
        # This prevents wasting compute when hosts are unreachable
        if HAS_PRE_SPAWN_HEALTH and gate_on_cluster_health is not None:
            try:
                can_proceed, health_msg = gate_on_cluster_health(
                    "training",
                    min_healthy=2,  # Need at least 2 hosts for distributed training
                    min_healthy_ratio=0.4,  # Allow training if 40%+ hosts healthy
                )
                if not can_proceed:
                    if self.config.verbose:
                        print(f"[Training] Deferred: {health_msg}")
                    return None
            except Exception as e:
                # Non-critical, proceed with training
                if self.config.verbose:
                    print(f"[Training] Cluster health check failed: {e}, proceeding anyway")

        now = time.time()

        # Get configs sorted by priority (underperforming configs first)
        # This ensures that when multiple configs are ready, we train the most urgent one
        priority_queue = ConfigPriorityQueue()
        prioritized_configs = priority_queue.get_prioritized_configs(self.state.configs)

        for config_key, priority_score in prioritized_configs:
            config_state = self.state.configs[config_key]
            # Check minimum interval between training runs
            if now - config_state.last_training_time < self.config.min_interval_seconds:
                continue

            # Trigger 0: Momentum-based acceleration (positive feedback optimization)
            # When a model is improving, train more frequently to capitalize on momentum
            if HAS_FEEDBACK_ACCELERATOR:
                try:
                    decision = get_feedback_accelerator().get_training_decision(config_key)
                    if decision.should_train:
                        # Update games_since_training in accelerator
                        record_games_generated(config_key, config_state.games_since_training)

                        intensity_str = decision.intensity.value if decision.intensity else "normal"
                        momentum_str = decision.momentum.value if decision.momentum else "stable"
                        print(f"[Training] Trigger: momentum-based acceleration for {config_key} "
                              f"(intensity={intensity_str}, momentum={momentum_str}, "
                              f"threshold={decision.min_games_threshold})")
                        return config_key
                except Exception as e:
                    if self.config.verbose:
                        print(f"[Training] Feedback accelerator error: {e}")

            # Trigger 1: Dynamic game count threshold (adapts to promotion velocity)
            dynamic_threshold = self._get_dynamic_threshold(config_key)
            if config_state.games_since_training >= dynamic_threshold:
                print(f"[Training] Trigger: game threshold reached for {config_key} "
                      f"({config_state.games_since_training} >= {dynamic_threshold} games)")
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

        # Record training start for dynamic threshold velocity tracking
        self.record_training_start()

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

            # Estimate training duration and log ETA
            if HAS_COORDINATION:
                est_duration = estimate_task_duration("training", config=config_key)
                eta_time = datetime.fromtimestamp(time.time() + est_duration).strftime("%H:%M:%S")
                print(f"[Training] Estimated duration: {est_duration/3600:.1f}h (ETA: {eta_time})")

            # Use v3 for all board types (best architecture with spatial policy heads)
            model_version = "v3"

            # Find game data - check both DB files and JSONL files (GPU selfplay)
            games_dir = AI_SERVICE_ROOT / "data" / "games"
            synced_dir = games_dir / "synced"
            gpu_selfplay_dir = games_dir / "gpu_selfplay"

            # Check for JSONL data for this config (preferred for GPU selfplay)
            jsonl_path = gpu_selfplay_dir / config_key / "games.jsonl"
            has_jsonl_data = jsonl_path.exists() and jsonl_path.stat().st_size > 0

            # Collect databases from main games dir and synced subdirectory
            game_dbs = list(games_dir.glob("*.db"))
            if synced_dir.exists():
                # Include all synced databases (cluster data)
                game_dbs.extend(synced_dir.rglob("*.db"))

            if not game_dbs and not has_jsonl_data:
                print(f"[Training] No game data found (DB: {games_dir}, JSONL: {jsonl_path})")
                self.state.training_in_progress = False
                self._release_training_lock()
                return False

            if has_jsonl_data:
                print(f"[Training] Found JSONL data for {config_key}: {jsonl_path}")

            # Auto-consolidate databases if consolidated DB is stale or missing
            # Skip consolidation if using JSONL data (will use jsonl_to_npz.py instead)
            consolidated_db = games_dir / "consolidated_training_v2.db"
            consolidation_max_age = 6 * 3600  # 6 hours

            should_consolidate = False
            if has_jsonl_data:
                # Using JSONL data - skip DB consolidation
                print(f"[Training] Using JSONL data, skipping DB consolidation")
            elif not consolidated_db.exists():
                should_consolidate = True
                print(f"[Training] Consolidated DB missing, will merge databases")
            elif time.time() - consolidated_db.stat().st_mtime > consolidation_max_age:
                should_consolidate = True
                print(f"[Training] Consolidated DB stale (>{consolidation_max_age/3600:.0f}h old), will refresh")

            if should_consolidate:
                # Find all selfplay DBs to merge (exclude quarantine, corrupted, etc.)
                merge_dbs = []
                skipped_invalid = 0
                for db_path in game_dbs:
                    db_str = str(db_path)
                    # Skip quarantine, corrupted, backup, and the consolidated DB itself
                    if any(skip in db_str for skip in ['quarantine', 'corrupted', 'backup', 'consolidated']):
                        continue
                    # Only include selfplay/merged databases
                    if 'selfplay' in db_path.name or 'merged' in db_path.name or 'training' in db_path.name:
                        # Pre-consolidation validation: check DB structure
                        try:
                            conn = sqlite3.connect(db_path)
                            cursor = conn.cursor()
                            # Check games table exists and has data
                            cursor.execute("SELECT COUNT(*) FROM games")
                            count = cursor.fetchone()[0]
                            # Check game_moves table exists
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'")
                            has_moves = cursor.fetchone() is not None
                            conn.close()

                            if count > 0 and has_moves:
                                merge_dbs.append(db_path)
                            else:
                                skipped_invalid += 1
                        except Exception:
                            skipped_invalid += 1

                if skipped_invalid > 0:
                    print(f"[Training] Skipped {skipped_invalid} invalid/empty databases")

                # Elo-based quality filter: get models with sufficient Elo via centralized service
                high_elo_models = set()
                min_model_elo = 1300  # Minimum Elo to include games from this model
                if get_elo_service is not None:
                    try:
                        elo_svc = get_elo_service()
                        rows = elo_svc.execute_query("""
                            SELECT participant_id, rating FROM elo_ratings
                            WHERE rating >= ? AND games_played >= 5
                        """, (min_model_elo,))
                        for row in rows:
                            high_elo_models.add(row[0])
                            # Also add without prefix variations
                            if row[0].startswith("nn:"):
                                high_elo_models.add(row[0][3:])
                        print(f"[Training] Elo filter: {len(high_elo_models)} models with Elo >= {min_model_elo}")
                    except Exception as e:
                        print(f"[Training] Elo filter skipped: {e}")

                if len(merge_dbs) > 1:
                    print(f"[Training] Consolidating {len(merge_dbs)} validated databases...")
                    merge_cmd = [
                        sys.executable,
                        str(AI_SERVICE_ROOT / "scripts" / "merge_game_dbs.py"),
                        "--output", str(consolidated_db),
                        "--dedupe-by-game-id",
                    ]
                    for db in merge_dbs:
                        merge_cmd.extend(["--db", str(db)])

                    try:
                        merge_process = await asyncio.create_subprocess_exec(
                            *merge_cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd=AI_SERVICE_ROOT,
                        )
                        stdout, stderr = await asyncio.wait_for(
                            merge_process.communicate(),
                            timeout=1800  # 30 min max for consolidation
                        )
                        if merge_process.returncode == 0:
                            print(f"[Training] Database consolidation complete")

                            # Generate parity fixtures for the consolidated DB
                            parity_fixtures_dir = AI_SERVICE_ROOT / "data" / "parity_fixtures"
                            parity_fixtures_dir.mkdir(parents=True, exist_ok=True)
                            print(f"[Training] Generating parity fixtures...")
                            parity_cmd = [
                                sys.executable,
                                str(AI_SERVICE_ROOT / "scripts" / "check_ts_python_replay_parity.py"),
                                "--db", str(consolidated_db),
                                "--emit-fixtures-dir", str(parity_fixtures_dir),
                                "--limit-games-per-db", "1000",  # Sample for speed
                            ]
                            try:
                                parity_process = await asyncio.create_subprocess_exec(
                                    *parity_cmd,
                                    stdout=asyncio.subprocess.PIPE,
                                    stderr=asyncio.subprocess.PIPE,
                                    cwd=AI_SERVICE_ROOT,
                                )
                                await asyncio.wait_for(
                                    parity_process.communicate(),
                                    timeout=600  # 10 min max for parity check
                                )
                                if parity_process.returncode == 0:
                                    print(f"[Training] Parity fixtures generated")
                            except Exception as e:
                                print(f"[Training] Parity fixtures generation skipped: {e}")
                        else:
                            print(f"[Training] Consolidation failed: {stderr.decode()[:500]}")
                    except asyncio.TimeoutError:
                        print(f"[Training] Consolidation timed out, using largest DB instead")
                    except Exception as e:
                        print(f"[Training] Consolidation error: {e}")

            # Prefer consolidated database if it exists, otherwise use largest
            if consolidated_db.exists():
                largest_db = consolidated_db
                print(f"[Training] Using consolidated database: {largest_db}")
            else:
                largest_db = max(game_dbs, key=lambda p: p.stat().st_size)
                print(f"[Training] Using largest database: {largest_db} ({largest_db.stat().st_size / 1024 / 1024:.1f}MB)")

            # Export training data with appropriate encoder
            training_dir = AI_SERVICE_ROOT / "data" / "training"
            training_dir.mkdir(parents=True, exist_ok=True)
            data_path = training_dir / f"unified_{config_key}.npz"

            # Skip export if NPZ file exists and is recent (less than 6 hours old)
            # This allows reuse of existing data and avoids errors from corrupted DBs
            export_max_age = 6 * 3600  # 6 hours
            skip_export = False
            if data_path.exists():
                npz_age = time.time() - data_path.stat().st_mtime
                if npz_age < export_max_age:
                    print(f"[Training] Skipping export - NPZ file recent ({npz_age/3600:.1f}h old < {export_max_age/3600:.0f}h)")
                    skip_export = True

            # Use JSONL export if we have JSONL data (checked earlier)
            use_jsonl_export = has_jsonl_data

            # Determine encoder version (v2 for hex to match 20 global features)
            encoder_version = "v2" if board_type == "hexagonal" else "default"

            if not skip_export:
                if use_jsonl_export:
                    # Use jsonl_to_npz.py for JSONL data (from GPU selfplay)
                    print(f"[Training] Using JSONL source: {jsonl_path}")
                    export_cmd = [
                        sys.executable,
                        str(AI_SERVICE_ROOT / "scripts" / "jsonl_to_npz.py"),
                        "--input", str(jsonl_path),
                        "--output", str(data_path),
                        "--board-type", board_type,
                        "--num-players", str(num_players),
                        "--gpu-selfplay",  # Flag for GPU selfplay format
                        "--max-games", "10000",  # Reasonable limit
                    ]
                    if encoder_version != "default":
                        export_cmd.extend(["--encoder-version", encoder_version])
                else:
                    # Use export_replay_dataset.py for DB data
                    export_cmd = [
                        sys.executable,
                        str(AI_SERVICE_ROOT / self.config.export_script),
                        "--db", str(largest_db),
                        "--output", str(data_path),
                        "--board-type", board_type,
                        "--num-players", str(num_players),
                        "--sample-every", "2",
                        # Quality filters - only train on good data
                        "--require-completed",  # Only games that completed normally
                        "--min-moves", "10",    # Filter out trivially short games
                        "--exclude-recovery",   # Exclude error recovery games
                    ]
                    if encoder_version != "default":
                        export_cmd.extend(["--encoder-version", encoder_version])

                # Use parity fixtures to truncate games at safe points (avoid parity divergence)
                parity_fixtures_dir = AI_SERVICE_ROOT / "data" / "parity_fixtures"
                if parity_fixtures_dir.exists() and any(parity_fixtures_dir.iterdir()):
                    export_cmd.extend(["--parity-fixtures-dir", str(parity_fixtures_dir)])
                    print(f"[Training] Using parity fixtures from {parity_fixtures_dir}")

                print(f"[Training] Exporting data for {config_key} (encoder: {encoder_version})...")
                # Set PYTHONPATH to AI_SERVICE_ROOT so that 'app' module can be imported
                export_env = os.environ.copy()
                export_env["PYTHONPATH"] = str(AI_SERVICE_ROOT)
                export_process = await asyncio.create_subprocess_exec(
                    *export_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=AI_SERVICE_ROOT,
                    env=export_env,
                )
                stdout, stderr = await export_process.communicate()

                if export_process.returncode != 0:
                    error_msg = stderr.decode().strip()[:500] if stderr else "Unknown error"
                    print(f"[Training] Export failed for {config_key}: {error_msg}")
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

            # Record training completion in improvement optimizer for positive feedback
            if HAS_IMPROVEMENT_OPTIMIZER and success:
                try:
                    optimizer = get_improvement_optimizer()
                    calibration_ece = None
                    if "calibration" in result:
                        calibration_ece = result["calibration"].get("ece")

                    rec = optimizer.record_training_complete(
                        config_key=config_key,
                        duration_seconds=result["duration"],
                        val_loss=0.0,  # Not tracked here, but could be added
                        calibration_ece=calibration_ece,
                    )
                    metrics = optimizer.get_improvement_metrics()
                    print(f"[ImprovementOptimizer] Training complete recorded for {config_key}: "
                          f"duration={result['duration']/60:.1f}min, "
                          f"training_runs_24h={metrics['training_runs_24h']}")
                except Exception as e:
                    if self.config.verbose:
                        print(f"[ImprovementOptimizer] Error recording training: {e}")

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

    async def request_urgent_training(self, configs: List[str], reason: str) -> bool:
        """Request urgent training for specified configs due to feedback signal.

        This bypasses normal game count thresholds when the feedback system
        detects repeated failures (e.g., consecutive promotion failures).

        Args:
            configs: List of config keys to train
            reason: Human-readable reason for urgent training

        Returns:
            True if training was started, False otherwise
        """
        if self.state.training_in_progress:
            print(f"[Training] Urgent training request deferred: training already in progress")
            return False

        # Pick the first config that hasn't been trained recently
        now = time.time()
        for config_key in configs:
            if config_key not in self.state.configs:
                continue

            config_state = self.state.configs[config_key]
            # Allow urgent training even if recently trained (use shorter cooldown)
            urgent_cooldown = self.config.min_interval_seconds / 2
            if now - config_state.last_training_time < urgent_cooldown:
                print(f"[Training] Urgent training for {config_key} still in cooldown")
                continue

            print(f"[Training] URGENT TRAINING triggered for {config_key}: {reason}")
            started = await self.start_training(config_key)
            if started:
                return True

        print(f"[Training] Urgent training request could not be fulfilled for configs: {configs}")
        return False


# =============================================================================
# Model Promoter Component
# EXTRACTED: Now imported from scripts.unified_loop.promotion (Phase 2 refactoring)
# =============================================================================

# =============================================================================
# Adaptive Curriculum Component
# EXTRACTED: Now imported from scripts.unified_loop.curriculum (Phase 2 refactoring)
# =============================================================================

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

        # Initialize cross-process event queue for bidirectional event bridging
        self._cross_process_queue: Optional[CrossProcessEventQueue] = None
        if HAS_CROSS_PROCESS_EVENTS:
            try:
                self._cross_process_queue = CrossProcessEventQueue()
                self.event_bus.set_cross_process_queue(self._cross_process_queue)
                print("[UnifiedLoop] Cross-process event publishing enabled")
            except Exception as e:
                print(f"[UnifiedLoop] Warning: Cross-process queue init failed: {e}")

        # Set up circuit breaker event publishing
        if HAS_CIRCUIT_BREAKER and self._cross_process_queue is not None:
            def on_circuit_state_change(target: str, old_state: CircuitState, new_state: CircuitState):
                """Publish circuit breaker state changes to cross-process queue."""
                try:
                    event_type = f"circuit_{new_state.value}"  # e.g., circuit_open, circuit_closed
                    self._cross_process_queue.publish(
                        event_type=event_type,
                        payload={
                            "target": target,
                            "old_state": old_state.value,
                            "new_state": new_state.value,
                            "transition": f"{old_state.value} -> {new_state.value}",
                        },
                        source="unified_ai_loop"
                    )
                    print(f"[CircuitBreaker] {target}: {old_state.value} -> {new_state.value}")
                except Exception as e:
                    print(f"[CircuitBreaker] Event publish error: {e}")

            set_host_breaker_callback(on_circuit_state_change)
            print("[UnifiedLoop] Circuit breaker event publishing enabled")

        # Initialize components
        self.data_collector = StreamingDataCollector(
            config.data_ingestion, self.state, self.event_bus
        )
        self.shadow_tournament = ShadowTournamentService(
            config.evaluation, self.state, self.event_bus
        )
        # Priority queue for focusing resources on underperforming configs
        self.config_priority = ConfigPriorityQueue()
        self.training_scheduler = TrainingScheduler(
            config.training, self.state, self.event_bus,
            feedback_config=config.feedback,  # Pass feedback config for performance-based triggers
            config_priority=self.config_priority  # Pass priority queue for dynamic thresholds
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
        self.model_pruning = ModelPruningService(
            config.model_pruning, self.state, self.event_bus
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
                # CMA-ES trigger handler
                self.event_bus.subscribe(DataEventType.CMAES_TRIGGERED, self._handle_cmaes_trigger)
                # Subscribe to failure events for coordinated error handling
                self.event_bus.subscribe(DataEventType.TRAINING_FAILED, self._on_training_failed)
                self.event_bus.subscribe(DataEventType.EVALUATION_FAILED, self._on_evaluation_failed)
                self.event_bus.subscribe(DataEventType.PROMOTION_FAILED, self._on_promotion_failed)
                self.event_bus.subscribe(DataEventType.DATA_SYNC_FAILED, self._on_data_sync_failed)
                print("[UnifiedLoop] Failure event handlers registered")
            except Exception as e:
                print(f"[UnifiedLoop] Warning: Failed to initialize feedback controller: {e}")

        # Feedback signal router for routing FeedbackAction signals to handlers
        self.feedback_router: Optional[FeedbackSignalRouter] = None
        if HAS_FEEDBACK and FeedbackSignalRouter is not None:
            try:
                self.feedback_router = FeedbackSignalRouter()

                # Register handlers for key feedback actions
                self.feedback_router.register_handler(
                    FeedbackAction.INCREASE_DATA_COLLECTION,
                    self._handle_increase_data_collection,
                    "unified_loop_increase_data"
                )
                self.feedback_router.register_handler(
                    FeedbackAction.URGENT_RETRAINING,
                    self._handle_urgent_retraining,
                    "unified_loop_urgent_retrain"
                )
                self.feedback_router.register_handler(
                    FeedbackAction.TRIGGER_CMAES,
                    self._handle_cmaes_feedback_signal,
                    "unified_loop_cmaes"
                )
                self.feedback_router.register_handler(
                    FeedbackAction.SCALE_UP_SELFPLAY,
                    self._handle_scale_up_selfplay,
                    "unified_loop_scale_up"
                )
                self.feedback_router.register_handler(
                    FeedbackAction.SCALE_DOWN_SELFPLAY,
                    self._handle_scale_down_selfplay,
                    "unified_loop_scale_down"
                )
                print("[UnifiedLoop] FeedbackSignalRouter initialized with handlers")
            except Exception as e:
                print(f"[UnifiedLoop] Warning: Failed to initialize feedback router: {e}")

        # Health monitoring
        global _health_tracker
        self.health_tracker = HealthTracker()
        _health_tracker = self.health_tracker  # Set global for HTTP endpoint
        print("[UnifiedLoop] Health tracker initialized")

        # Task coordination - prevents runaway spawning across orchestrators
        self.task_coordinator: Optional[TaskCoordinator] = None
        self._task_id: Optional[str] = None  # Initialize to avoid AttributeError on shutdown
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
        self._p2p_backend: Optional[P2PBackend] = None  # Direct backend for low-level API calls
        if HAS_P2P and config.p2p.enabled:
            try:
                # Try resilient leader discovery if seed URLs configured
                effective_base_url = config.p2p.p2p_base_url
                p2p_seed_urls = os.environ.get("RINGRIFT_P2P_SEEDS", "").split(",")
                p2p_seed_urls = [s.strip() for s in p2p_seed_urls if s.strip()]

                if HAS_P2P_BACKEND and discover_p2p_leader_url and p2p_seed_urls:
                    # Use resilient leader discovery
                    try:
                        discovered_url = asyncio.get_event_loop().run_until_complete(
                            discover_p2p_leader_url(
                                p2p_seed_urls,
                                auth_token=config.p2p.auth_token or "",
                            )
                        )
                        if discovered_url:
                            effective_base_url = discovered_url
                            print(f"[UnifiedLoop] P2P leader discovered: {effective_base_url}")
                    except Exception as disc_err:
                        print(f"[UnifiedLoop] P2P leader discovery failed, using config URL: {disc_err}")

                # Create P2PIntegrationConfig from our local config
                p2p_config = P2PIntegrationConfig(
                    p2p_base_url=effective_base_url,
                    auth_token=config.p2p.auth_token,
                    model_sync_enabled=config.p2p.model_sync_enabled,
                    target_selfplay_games_per_hour=config.p2p.target_selfplay_games_per_hour,
                    auto_scale_selfplay=config.p2p.auto_scale_selfplay,
                    use_distributed_tournament=config.p2p.use_distributed_tournament,
                    health_check_interval=config.p2p.health_check_interval,
                    sync_interval_seconds=config.p2p.sync_interval_seconds,
                )
                self.p2p = P2PIntegrationManager(p2p_config)

                # Also create a P2PBackend for direct API access
                if HAS_P2P_BACKEND and P2PBackend is not None:
                    self._p2p_backend = P2PBackend(
                        effective_base_url,
                        auth_token=config.p2p.auth_token,
                    )

                # Wire P2P callbacks to our event bus
                self.p2p.register_callback("cluster_unhealthy", self._on_p2p_cluster_unhealthy)
                self.p2p.register_callback("nodes_dead", self._on_p2p_nodes_dead)
                self.p2p.register_callback("selfplay_scaled", self._on_p2p_selfplay_scaled)

                print(f"[UnifiedLoop] P2P integration initialized (base_url={effective_base_url})")
            except Exception as e:
                print(f"[UnifiedLoop] Warning: Failed to initialize P2P integration: {e}")
                self.p2p = None

        # Execution backend - unified interface for local/SSH/P2P execution
        self.backend: Optional[OrchestratorBackend] = None
        if HAS_BACKENDS:
            try:
                # Auto-detect backend type from config
                backend_type = None
                if config.p2p.enabled and self.p2p is not None:
                    # P2P available - could use P2P backend in future
                    pass  # For now, fall through to SSH/Local detection
                # get_backend auto-detects SSH if hosts config exists, else Local
                self.backend = get_backend(backend_type, force_new=True)
                print(f"[UnifiedLoop] Execution backend initialized ({type(self.backend).__name__})")
            except Exception as e:
                print(f"[UnifiedLoop] Warning: Failed to initialize execution backend: {e}")

        # Resource optimizer - cooperative cluster-wide utilization targeting (60-80%)
        self.resource_optimizer: Optional[ResourceOptimizer] = None
        if HAS_RESOURCE_OPTIMIZER:
            try:
                self.resource_optimizer = get_resource_optimizer()
                # Set orchestrator ID for tracking (os is imported globally)
                os.environ.setdefault("RINGRIFT_ORCHESTRATOR", "unified_ai_loop")
                print("[UnifiedLoop] Resource optimizer initialized (target: 60-80% utilization)")
            except Exception as e:
                print(f"[UnifiedLoop] Warning: Failed to initialize resource optimizer: {e}")

        # Priority job scheduler - ensures critical jobs (promotion eval) preempt selfplay
        self.job_scheduler: Optional[PriorityJobScheduler] = None
        if HAS_JOB_SCHEDULER:
            try:
                self.job_scheduler = get_job_scheduler()
                print("[UnifiedLoop] Priority job scheduler initialized (CRITICAL > HIGH > NORMAL > LOW)")
            except Exception as e:
                print(f"[UnifiedLoop] Warning: Failed to initialize job scheduler: {e}")

        # Stage event bus - pipeline stage completion events for cross-component coordination
        self.stage_event_bus: Optional[StageEventBus] = None
        if HAS_STAGE_EVENTS:
            try:
                self.stage_event_bus = get_stage_event_bus()
                # Bridge stage events to the main event bus
                self._setup_stage_event_bridge()
                print("[UnifiedLoop] Stage event bus initialized and bridged to main EventBus")
            except Exception as e:
                print(f"[UnifiedLoop] Warning: Failed to initialize stage event bus: {e}")

        # Unified promotion controller - augments ModelPromoter with consistent criteria
        self.promotion_controller: Optional[PromotionController] = None
        if HAS_PROMOTION_CONTROLLER:
            try:
                criteria = PromotionCriteria(
                    min_elo_improvement=config.promotion.min_elo_improvement,
                    min_games_played=config.promotion.min_games,
                    min_win_rate=config.promotion.min_win_rate,
                    confidence_threshold=config.promotion.statistical_significance,
                )
                self.promotion_controller = PromotionController(criteria=criteria)
                print(f"[UnifiedLoop] Promotion controller initialized (min_elo={criteria.min_elo_improvement})")
            except Exception as e:
                print(f"[UnifiedLoop] Warning: Failed to initialize promotion controller: {e}")

        # State management
        self._state_path = AI_SERVICE_ROOT / config.log_dir / "unified_loop_state.json"
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Timing trackers
        self._last_shadow_eval: Dict[str, float] = {}
        self._last_full_eval: float = 0.0
        self._last_diverse_tournament: float = 0.0
        self._started_time: float = 0.0

        # Gauntlet evaluation and model culling
        self.gauntlet: Optional[DistributedNNGauntlet] = None
        self.culler: Optional[ModelCullingController] = None
        self._last_gauntlet_check: float = 0.0
        self._gauntlet_interval: float = 1800.0  # 30 minutes between gauntlet checks
        if HAS_GAUNTLET:
            try:
                elo_db_path = AI_SERVICE_ROOT / "data" / "unified_elo.db"
                model_dir = AI_SERVICE_ROOT / "data" / "models"
                self.gauntlet = get_gauntlet(elo_db_path, model_dir)
                self.culler = get_culling_controller(elo_db_path, model_dir)
                print("[UnifiedLoop] Gauntlet evaluator and model culler initialized")
            except Exception as e:
                print(f"[UnifiedLoop] Warning: Failed to initialize gauntlet/culler: {e}")

        # Elo database synchronization for cluster-wide consistency
        self.elo_sync_manager: Optional[EloSyncManager] = None
        self._elo_sync_interval: float = 300.0  # 5 minutes
        self._last_elo_sync: float = 0.0
        if HAS_ELO_SYNC:
            try:
                elo_db_path = AI_SERVICE_ROOT / "data" / "unified_elo.db"
                self.elo_sync_manager = get_elo_sync_manager(
                    db_path=elo_db_path,
                    coordinator_host="lambda-h100"
                )
                print("[UnifiedLoop] EloSyncManager initialized for cluster-wide Elo consistency")
            except Exception as e:
                print(f"[UnifiedLoop] Warning: Failed to initialize EloSyncManager: {e}")

    def _setup_stage_event_bridge(self):
        """Bridge StageEventBus events to the main EventBus for unified coordination.

        Maps stage events to internal DataEventType events so both event systems
        work together seamlessly.
        """
        if not HAS_STAGE_EVENTS or self.stage_event_bus is None:
            return

        async def on_selfplay_complete(result: StageCompletionResult):
            """Bridge selfplay completion to main event bus."""
            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.DATA_SYNC_COMPLETED,
                payload={
                    "games_synced": result.games_generated,
                    "iteration": result.iteration,
                    "config": f"{result.board_type}_{result.num_players}p",
                    "source": "stage_event_bus",
                }
            ))

        async def on_training_complete(result: StageCompletionResult):
            """Bridge training completion to main event bus."""
            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.TRAINING_COMPLETED,
                payload={
                    "config_key": f"{result.board_type}_{result.num_players}p",
                    "model_path": result.model_path,
                    "train_loss": result.train_loss,
                    "val_loss": result.val_loss,
                    "source": "stage_event_bus",
                }
            ))

        async def on_evaluation_complete(result: StageCompletionResult):
            """Bridge evaluation completion to main event bus."""
            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.EVALUATION_COMPLETED,
                payload={
                    "config_key": f"{result.board_type}_{result.num_players}p",
                    "win_rate": result.win_rate,
                    "elo_delta": result.elo_delta,
                    "source": "stage_event_bus",
                }
            ))

        async def on_promotion_complete(result: StageCompletionResult):
            """Bridge promotion completion to main event bus."""
            if result.promoted:
                await self.event_bus.publish(DataEvent(
                    event_type=DataEventType.MODEL_PROMOTED,
                    payload={
                        "config": f"{result.board_type}_{result.num_players}p",
                        "model_id": result.model_id,
                        "elo_gain": result.elo_delta,
                        "reason": result.promotion_reason,
                        "source": "stage_event_bus",
                    }
                ))

        # Subscribe to stage events
        self.stage_event_bus.subscribe(StageEvent.SELFPLAY_COMPLETE, on_selfplay_complete)
        self.stage_event_bus.subscribe(StageEvent.TRAINING_COMPLETE, on_training_complete)
        self.stage_event_bus.subscribe(StageEvent.EVALUATION_COMPLETE, on_evaluation_complete)
        self.stage_event_bus.subscribe(StageEvent.PROMOTION_COMPLETE, on_promotion_complete)

    async def _emit_stage_event(
        self,
        event: "StageEvent",
        success: bool,
        **kwargs,
    ):
        """Emit a stage completion event to the StageEventBus.

        Args:
            event: The StageEvent type to emit
            success: Whether the stage completed successfully
            **kwargs: Additional fields for StageCompletionResult
        """
        if not HAS_STAGE_EVENTS or self.stage_event_bus is None:
            return

        result = StageCompletionResult(
            event=event,
            success=success,
            iteration=self.state.iteration,
            timestamp=datetime.now().isoformat(),
            board_type=kwargs.get("board_type", "square8"),
            num_players=kwargs.get("num_players", 2),
            games_generated=kwargs.get("games_generated", 0),
            model_path=kwargs.get("model_path"),
            model_id=kwargs.get("model_id"),
            train_loss=kwargs.get("train_loss"),
            val_loss=kwargs.get("val_loss"),
            win_rate=kwargs.get("win_rate"),
            elo_delta=kwargs.get("elo_delta"),
            promoted=kwargs.get("promoted", False),
            promotion_reason=kwargs.get("promotion_reason"),
            error=kwargs.get("error"),
            metadata=kwargs.get("metadata", {}),
        )

        await self.stage_event_bus.emit(result)

    def _schedule_job(
        self,
        job_type: str,
        priority: "JobPriority",
        config: Dict[str, Any],
        requires_gpu: bool = False,
        host_preference: Optional[str] = None,
    ) -> bool:
        """Schedule a job with the priority job scheduler.

        Args:
            job_type: Type of job (selfplay, training, evaluation, promotion)
            priority: Job priority level
            config: Job configuration
            requires_gpu: Whether job needs GPU
            host_preference: Preferred host for the job

        Returns:
            True if job was scheduled successfully
        """
        if not HAS_JOB_SCHEDULER or self.job_scheduler is None:
            return True  # No scheduler, proceed directly

        job = ScheduledJob(
            job_type=job_type,
            priority=priority,
            config=config,
            requires_gpu=requires_gpu,
            host_preference=host_preference,
        )
        return self.job_scheduler.schedule(job)

    def _get_next_scheduled_job(self) -> Optional[Tuple["ScheduledJob", Any]]:
        """Get the next job to run from the priority queue.

        Returns:
            (job, host) tuple if a job is ready, None otherwise
        """
        if not HAS_JOB_SCHEDULER or self.job_scheduler is None:
            return None

        # Build host list from state
        hosts = []
        statuses = []
        for name, host_state in self.state.hosts.items():
            if host_state.enabled and host_state.consecutive_failures < 3:
                hosts.append(host_state)
                statuses.append(host_state)

        return self.job_scheduler.next_job(
            hosts,
            statuses,
            host_get_name=lambda h: h.name if hasattr(h, 'name') else str(h),
            host_has_gpu=lambda h: getattr(h, 'has_gpu', False),
            host_get_memory_gb=lambda h: getattr(h, 'memory_gb', 0),
            status_get_cpu=lambda s: getattr(s, 'cpu_percent', 0.0),
            status_get_disk=lambda s: getattr(s, 'disk_percent', 0.0),
            status_get_memory=lambda s: getattr(s, 'memory_percent', 0.0),
            status_is_reachable=lambda s: getattr(s, 'reachable', True),
        )

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

        # Update cluster utilization metrics (feedback from P2P orchestrator)
        self._update_cluster_utilization_metrics()

    def _update_cluster_utilization_metrics(self):
        """Update metrics from unified resource targets (P2P feedback loop).

        Integrates with both resource_targets (per-host decisions) and
        resource_optimizer (cluster-wide PID-controlled optimization) to
        maintain 60-80% CPU/GPU utilization for optimal training throughput.
        """
        if not HAS_COORDINATION:
            return

        try:
            summary = get_cluster_summary()
            targets = get_resource_targets()
            now = time.time()

            # Initialize state tracking for hysteresis
            if not hasattr(self, '_last_utilization_log'):
                self._last_utilization_log = 0.0
                self._last_backpressure_change = 0.0

            avg_cpu = summary['avg_cpu']
            avg_gpu = summary['avg_gpu']
            avg_memory = summary['avg_memory']
            total_jobs = summary['total_jobs']
            current_bp = summary['backpressure_factor']
            active_hosts = summary['active_hosts']

            # Update Prometheus metrics for monitoring dashboards
            if HAS_PROMETHEUS:
                try:
                    CLUSTER_CPU_UTILIZATION.set(avg_cpu)
                    CLUSTER_GPU_UTILIZATION.set(avg_gpu)
                    CLUSTER_MEMORY_UTILIZATION.set(avg_memory)
                    CLUSTER_TOTAL_JOBS.set(total_jobs)
                    CLUSTER_BACKPRESSURE.set(current_bp)
                except Exception:
                    pass  # Gauges may not be registered in all environments

            # Determine cluster status
            if active_hosts > 0:
                if avg_cpu < targets.cpu_min and avg_gpu < targets.gpu_min:
                    status = "UNDERUTILIZED"
                elif avg_cpu > targets.cpu_max or avg_gpu > targets.gpu_max:
                    status = "OVERLOADED"
                else:
                    status = "OPTIMAL"
            else:
                status = "NO_DATA"

            # Periodic logging (every 5 minutes, or always if verbose)
            should_log = self.config.verbose or (now - self._last_utilization_log > 300)
            if should_log and active_hosts > 0:
                print(f"[Utilization] {status} | "
                      f"Hosts: {active_hosts}, Jobs: {total_jobs} | "
                      f"CPU: {avg_cpu:.1f}% (target {targets.cpu_min:.0f}-{targets.cpu_max:.0f}%) | "
                      f"GPU: {avg_gpu:.1f}% (target {targets.gpu_min:.0f}-{targets.gpu_max:.0f}%) | "
                      f"Backpressure: {current_bp:.2f}")
                self._last_utilization_log = now

            # Skip backpressure adjustment if no active hosts
            if active_hosts == 0:
                return

            # Hysteresis band to prevent oscillation (5% outside target range)
            hysteresis = 5.0
            new_bp = current_bp

            # Check if cluster is significantly underutilized (below min - hysteresis)
            if avg_cpu < (targets.cpu_min - hysteresis) and avg_gpu < (targets.gpu_min - hysteresis):
                # Relax backpressure gradually to allow more production
                new_bp = min(1.0, current_bp + 0.1)

            # Check if cluster is overloaded (above max + hysteresis)
            elif avg_cpu > (targets.cpu_max + hysteresis) or avg_gpu > (targets.gpu_max + hysteresis):
                # Apply backpressure to prevent overload
                if avg_cpu > targets.cpu_critical or avg_gpu > targets.gpu_critical:
                    new_bp = max(0.5, current_bp - 0.2)  # Strong reduction
                else:
                    new_bp = max(0.7, current_bp - 0.1)  # Mild reduction

            # Only apply change if significant and not too frequent (60s cooldown)
            if abs(new_bp - current_bp) >= 0.05 and (now - self._last_backpressure_change > 60):
                set_backpressure(new_bp)
                self._last_backpressure_change = now
                if self.config.verbose:
                    print(f"[Utilization] Backpressure adjusted: {current_bp:.2f} -> {new_bp:.2f}")

            # Use resource optimizer for PID-controlled recommendations
            if HAS_RESOURCE_OPTIMIZER and self.resource_optimizer:
                try:
                    recommendation = self.resource_optimizer.get_optimization_recommendation()

                    # Log significant recommendations (60-80% target range)
                    if recommendation.action.value != "none" and recommendation.confidence > 0.3:
                        in_target = targets.cpu_min <= avg_cpu <= targets.cpu_max
                        opt_status = "IN TARGET" if in_target else "OUT OF TARGET"
                        print(f"[ResourceOptimizer] {opt_status} | {recommendation.action.value.upper()}: "
                              f"{recommendation.reason} (confidence: {recommendation.confidence:.0%})")

                except Exception as e:
                    if self.config.verbose:
                        print(f"[ResourceOptimizer] Error getting recommendation: {e}")

        except Exception as e:
            if self.config.verbose:
                print(f"[Utilization] Error getting cluster summary: {e}")

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

            # Update Elo and win rates from Elo database via centralized service
            if get_elo_service is not None:
                try:
                    elo_svc = get_elo_service()

                    # Get best Elo for each config (both ringrift_* and *_nn_baseline* models)
                    rows = elo_svc.execute_query("""
                        SELECT board_type, num_players, participant_id, rating, games_played
                        FROM elo_ratings
                        WHERE (participant_id LIKE 'ringrift_%' OR participant_id LIKE '%_nn_baseline%')
                        ORDER BY board_type, num_players, rating DESC
                    """)

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
                except Exception as e:
                    print(f"[Metrics] Elo service query failed: {e}")

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

    async def _trigger_elo_sync_after_evaluation(self, num_matches: int = 1):
        """Trigger ELO sync after evaluation to ensure cluster consistency.

        This is called after evaluation completion to ensure other nodes
        receive updated ELO ratings promptly. Debounces to avoid sync storms.
        """
        if not HAS_ELO_SYNC or self.elo_sync_manager is None:
            return

        MIN_MATCHES_FOR_IMMEDIATE_SYNC = 10
        MIN_INTERVAL_BETWEEN_SYNCS = 30  # seconds

        try:
            last_sync = getattr(self, '_last_eval_elo_sync', 0)
            now = time.time()

            # Accumulate pending matches
            pending = getattr(self, '_pending_eval_sync_matches', 0) + num_matches
            self._pending_eval_sync_matches = pending

            # Check if we should sync now
            should_sync = (
                pending >= MIN_MATCHES_FOR_IMMEDIATE_SYNC or
                (now - last_sync) >= MIN_INTERVAL_BETWEEN_SYNCS
            )

            if should_sync:
                self._last_eval_elo_sync = now
                self._pending_eval_sync_matches = 0
                self._last_elo_sync = now  # Update main sync timer too
                success = await self.elo_sync_manager.sync_with_cluster()
                if success:
                    print(f"[EloSync] Triggered after evaluation: "
                          f"{self.elo_sync_manager.state.local_match_count} matches")
        except Exception as e:
            print(f"[EloSync] Trigger after evaluation error: {e}")

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

            # Trigger ELO sync to propagate updated ratings to cluster
            num_matches = payload.get('games_played', 1)
            asyncio.create_task(self._trigger_elo_sync_after_evaluation(num_matches))

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
            games_used = payload.get('games_used', 0)
            initial_loss = payload.get('initial_loss', float('inf'))

            # Record in feedback accelerator for momentum tracking
            if HAS_FEEDBACK_ACCELERATOR and config_key:
                try:
                    # Determine if loss improved
                    loss_improved = (final_loss is not None and
                                     final_loss < initial_loss)

                    record_training_complete(
                        config_key=config_key,
                        loss_improved=loss_improved,
                        games_used=games_used
                    )
                    accelerator = get_feedback_accelerator()
                    momentum_data = accelerator.get_config_momentum(config_key)
                    if momentum_data:
                        print(f"[FeedbackAccelerator] Training complete for {config_key}: "
                              f"momentum={momentum_data.momentum_state.value}, "
                              f"intensity={momentum_data.intensity.value}")
                except Exception as e:
                    if self.config.verbose:
                        print(f"[FeedbackAccelerator] Error recording training complete: {e}")

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

            # Check for data collection adjustment signals and act on them
            pending_actions = self.feedback.get_pending_actions()
            for signal in pending_actions:
                if signal.action == FeedbackAction.INCREASE_DATA_COLLECTION:
                    print(f"[Feedback] INCREASE_DATA_COLLECTION signal: {signal.reason}")
                    await self._adjust_selfplay_rate(
                        multiplier=self.feedback.state.games_per_worker_multiplier,
                        reason=signal.reason
                    )
                elif signal.action == FeedbackAction.DECREASE_DATA_COLLECTION:
                    print(f"[Feedback] DECREASE_DATA_COLLECTION signal: {signal.reason}")
                    await self._adjust_selfplay_rate(
                        multiplier=self.feedback.state.games_per_worker_multiplier,
                        reason=signal.reason
                    )

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
            new_elo = payload.get('new_elo', 0)
            success = payload.get('success', True)
            reason = payload.get('reason', '')

            # Record in feedback accelerator for momentum tracking
            if HAS_FEEDBACK_ACCELERATOR and config_key:
                try:
                    if success and new_elo > 0:
                        # Record promotion to accelerate positive feedback
                        record_promotion(config_key, new_elo, model_id)
                        print(f"[FeedbackAccelerator] Recorded promotion for {config_key}: "
                              f"Elo={new_elo:.0f}")
                except Exception as e:
                    if self.config.verbose:
                        print(f"[FeedbackAccelerator] Error recording promotion: {e}")

            # Record in improvement optimizer for positive feedback amplification
            if HAS_IMPROVEMENT_OPTIMIZER and config_key:
                try:
                    optimizer = get_improvement_optimizer()
                    if success and elo_gain > 0:
                        rec = optimizer.record_promotion_success(
                            config_key=config_key,
                            elo_gain=elo_gain,
                            model_id=model_id or "",
                        )
                        metrics = optimizer.get_improvement_metrics()
                        print(f"[ImprovementOptimizer] Promotion success recorded for {config_key}: "
                              f"+{elo_gain:.0f} Elo, streak={metrics['consecutive_promotions']}, "
                              f"threshold→{metrics['effective_threshold']}")
                    else:
                        optimizer.record_promotion_failure(config_key, reason=reason)
                        print(f"[ImprovementOptimizer] Promotion failure recorded for {config_key}")
                except Exception as e:
                    if self.config.verbose:
                        print(f"[ImprovementOptimizer] Error recording promotion: {e}")

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

            # Check for urgent retraining or CMA-ES signals and act on them
            pending_actions = self.feedback.get_pending_actions()
            for signal in pending_actions:
                if signal.action == FeedbackAction.URGENT_RETRAINING:
                    print(f"[Feedback] URGENT RETRAINING signal: {signal.reason}")
                    # Get affected configs from signal metadata
                    affected_configs = signal.metadata.get('configs_affected', [])
                    if not affected_configs and config_key:
                        affected_configs = [config_key]
                    if affected_configs:
                        # Trigger urgent training through the scheduler
                        await self.training_scheduler.request_urgent_training(
                            configs=affected_configs,
                            reason=signal.reason
                        )
                elif signal.action == FeedbackAction.TRIGGER_CMAES:
                    print(f"[Feedback] CMA-ES trigger after promotion failures: {signal.reason}")
                    # Publish event to trigger CMA-ES optimization
                    await self.event_bus.publish(DataEvent(
                        event_type=DataEventType.CMAES_TRIGGERED,
                        payload={
                            "reason": signal.reason,
                            "source": "promotion_feedback",
                            "configs": [config_key] if config_key else [],
                        }
                    ))

        except Exception as e:
            print(f"[Feedback] Error processing promotion feedback: {e}")

    async def _on_elo_significant_change(self, event: DataEvent):
        """Handle significant Elo changes by triggering curriculum rebalancing.

        This enables event-driven curriculum rebalancing when a model's Elo
        changes significantly (exceeds the threshold in unified config).

        Also records Elo updates in the feedback accelerator for momentum tracking
        to enable positive feedback optimization.
        """
        try:
            payload = event.payload
            config_key = payload.get('config', '')
            model_id = payload.get('model_id', '')
            elo_change = payload.get('elo_change', 0)
            new_elo = payload.get('new_elo', 0)
            games_played = payload.get('games_played', 0)
            threshold = payload.get('threshold', 50)

            print(f"[Curriculum] Significant Elo change detected for {config_key}: "
                  f"{elo_change:+.1f} Elo (model: {model_id}, threshold: {threshold})")

            # Record Elo update in feedback accelerator for momentum tracking
            if HAS_FEEDBACK_ACCELERATOR and config_key and new_elo > 0:
                try:
                    momentum = record_elo_update(config_key, new_elo, games_played, model_id)
                    if momentum:
                        accelerator = get_feedback_accelerator()
                        momentum_data = accelerator.get_config_momentum(config_key)
                        if momentum_data:
                            print(f"[FeedbackAccelerator] {config_key}: Elo={new_elo:.0f}, "
                                  f"momentum={momentum_data.momentum_state.value}, "
                                  f"intensity={momentum_data.intensity.value}")
                except Exception as e:
                    if self.config.verbose:
                        print(f"[FeedbackAccelerator] Error recording Elo update: {e}")

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

    async def _adjust_selfplay_rate(self, multiplier: float, reason: str) -> None:
        """Adjust selfplay rate based on feedback signals.

        This method bridges the feedback controller's games_per_worker_multiplier
        to the P2P selfplay coordinator for distributed scaling. The rate is also
        persisted via resource_optimizer for cluster-wide coordination.

        Args:
            multiplier: Rate multiplier (1.0 = no change, 1.5 = 50% increase)
            reason: Human-readable reason for adjustment
        """
        new_rate = None

        # Update P2P coordinator if available
        if self.p2p and hasattr(self.p2p, 'selfplay'):
            try:
                new_rate = self.p2p.selfplay.adjust_target_rate(multiplier, reason)
                print(f"[Selfplay] Target rate adjusted to {new_rate}/hour (multiplier={multiplier:.2f})")

                # Emit event for coordination
                await self.event_bus.publish(DataEvent(
                    event_type=DataEventType.P2P_SELFPLAY_SCALED,
                    payload={
                        "new_rate": new_rate,
                        "multiplier": multiplier,
                        "reason": reason,
                    }
                ))
            except Exception as e:
                print(f"[Selfplay] Failed to adjust P2P target rate: {e}")
        else:
            # No P2P coordinator - just log the requested adjustment
            print(f"[Selfplay] Rate adjustment requested (multiplier={multiplier:.2f}): {reason}")
            print(f"[Selfplay] Note: P2P coordinator not available, adjustment not applied")

        # Persist via resource_optimizer for cluster-wide rate negotiation
        # This allows P2P orchestrator to query the negotiated rate
        if HAS_RESOURCE_OPTIMIZER and negotiate_selfplay_rate is not None:
            try:
                current = get_current_selfplay_rate() if get_current_selfplay_rate else 1000
                requested_rate = int(current * multiplier)
                approved_rate = negotiate_selfplay_rate(
                    requested_rate=requested_rate,
                    reason=f"unified_loop: {reason}",
                    requestor="unified_ai_loop",
                )
                if new_rate is None:
                    new_rate = approved_rate
                print(f"[Selfplay] Rate negotiated via resource_optimizer: {approved_rate}/hour")
            except Exception as e:
                print(f"[Selfplay] Resource optimizer rate negotiation failed: {e}")

        # Consult feedback accelerator for momentum-based rate recommendation
        # This allows the system to accelerate selfplay when models are improving
        if HAS_FEEDBACK_ACCELERATOR and get_aggregate_selfplay_recommendation is not None:
            try:
                # Get momentum-based recommendation across all configs
                recommendation = get_aggregate_selfplay_recommendation()
                if recommendation and recommendation.get('recommended_multiplier', 1.0) != 1.0:
                    rec_multiplier = recommendation['recommended_multiplier']
                    rec_reason = recommendation.get('reason', 'momentum-based adjustment')
                    momentum_state = recommendation.get('aggregate_momentum', 'unknown')

                    # Only log if momentum suggests a different rate than requested
                    if abs(rec_multiplier - multiplier) > 0.1:
                        print(f"[FeedbackAccelerator] Momentum-based rate recommendation: "
                              f"{rec_multiplier:.2f}x (aggregate momentum: {momentum_state})")

                        # If we're improving and the recommendation is higher, factor it in
                        if rec_multiplier > multiplier and momentum_state in ('accelerating', 'improving'):
                            final_multiplier = (multiplier + rec_multiplier) / 2
                            print(f"[FeedbackAccelerator] Blending momentum into rate: "
                                  f"{multiplier:.2f}x -> {final_multiplier:.2f}x")
            except Exception as e:
                if self.config.verbose:
                    print(f"[FeedbackAccelerator] Error getting rate recommendation: {e}")

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
    # Tier Gating (Difficulty Progression)
    # =========================================================================

    # Tier progression: D2 -> D4 -> D6 -> D8 (MCTS depth / difficulty)
    TIER_PROGRESSION = {
        "D2": "D4",
        "D4": "D6",
        "D6": "D8",
        "D8": "D8",  # Max tier
    }

    async def check_tier_promotion(
        self,
        config_key: str,
        win_rate_threshold: float = 0.55,
    ) -> Tuple[bool, str]:
        """Check if a model should be promoted to the next difficulty tier.

        Promotion requires winning > threshold against current tier opponents.
        Returns (should_promote, new_tier).
        """
        current_tier = self.state.tier_assignments.get(config_key, "D2")

        if current_tier not in self.TIER_PROGRESSION:
            return False, current_tier

        next_tier = self.TIER_PROGRESSION[current_tier]
        if next_tier == current_tier:
            print(f"[TierGating] {config_key} already at max tier {current_tier}")
            return False, current_tier

        # Check recent evaluation results for this config
        config_state = self.state.configs.get(config_key)
        if not config_state:
            print(f"[TierGating] No config state for {config_key}, skipping tier check")
            return False, current_tier

        # Use current Elo to estimate win rate against tier benchmark
        # Tier benchmarks: D2=1400, D4=1500, D6=1600, D8=1700
        tier_elo_benchmarks = {"D2": 1400, "D4": 1500, "D6": 1600, "D8": 1700}
        current_elo = config_state.current_elo
        tier_benchmark_elo = tier_elo_benchmarks.get(current_tier, 1500)

        # Estimate win rate: 1 / (1 + 10^((benchmark - elo) / 400))
        estimated_win_rate = 1.0 / (1.0 + 10 ** ((tier_benchmark_elo - current_elo) / 400))

        if estimated_win_rate >= win_rate_threshold:
            print(f"[TierGating] {config_key} promoted from {current_tier} to {next_tier} "
                  f"(est. win rate: {estimated_win_rate:.1%}, Elo: {current_elo:.0f})")
            self.state.tier_assignments[config_key] = next_tier
            self.state.tier_promotions_count += 1
            self._save_state()

            # Publish tier promotion event
            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.MODEL_PROMOTED,  # Reuse promotion event
                payload={
                    "type": "tier_promotion",
                    "config": config_key,
                    "old_tier": current_tier,
                    "new_tier": next_tier,
                    "win_rate": estimated_win_rate,
                    "elo": current_elo,
                }
            ))

            return True, next_tier

        print(f"[TierGating] {config_key} remains at {current_tier} "
              f"(est. win rate: {estimated_win_rate:.1%} < {win_rate_threshold:.1%})")
        return False, current_tier

    async def run_tier_gating_checks(self) -> Dict[str, str]:
        """Run tier gating checks for all configurations.

        Returns dict of config -> new_tier for any promotions.
        """
        print("[TierGating] === Running Tier Gating Checks ===")

        promotions = {}
        for config_key in self.state.configs:
            promoted, new_tier = await self.check_tier_promotion(config_key)
            if promoted:
                promotions[config_key] = new_tier

        if promotions:
            print(f"[TierGating] Tier promotions: {promotions}")
        else:
            print("[TierGating] No tier promotions this iteration")

        self.state.last_tier_check = time.time()
        return promotions

    def get_tier_for_config(self, config_key: str) -> str:
        """Get the current difficulty tier for a configuration."""
        return self.state.tier_assignments.get(config_key, "D2")

    # =========================================================================
    # Parity Validation Gate
    # =========================================================================

    async def run_parity_validation(
        self,
        board_type: str = "square8",
        num_players: int = 2,
    ) -> Dict[str, Any]:
        """Run parity validation gate on game databases.

        Validates that games pass the canonical parity check (TS engine replay
        produces identical states to Python). Only games that pass validation
        are used for training.

        Returns:
            Dict with validation results
        """
        print("[ParityValidation] === Starting Parity Validation Phase ===")

        results = {
            "total_games_checked": 0,
            "games_passed": 0,
            "games_failed": 0,
            "passed": False,
            "failure_rate": 0.0,
        }

        # Find parity validation script
        validation_script = AI_SERVICE_ROOT / "scripts" / "run_parity_validation.py"
        if not validation_script.exists():
            print("[ParityValidation] Validation script not found, skipping")
            results["passed"] = True  # Skip validation if script not present
            return results

        # Find game databases
        games_dir = AI_SERVICE_ROOT / "data" / "games"
        db_pattern = f"canonical_{board_type}_{num_players}p_*.db"
        game_dbs = list(games_dir.glob(db_pattern))

        if not game_dbs:
            # Try without canonical prefix
            db_pattern = f"*{board_type}*.db"
            game_dbs = list(games_dir.glob(db_pattern))

        if not game_dbs:
            print("[ParityValidation] No game databases found, skipping")
            results["passed"] = True
            return results

        # Run validation on largest DB
        largest_db = max(game_dbs, key=lambda p: p.stat().st_size)
        output_json = AI_SERVICE_ROOT / "data" / "parity_validation_results.json"

        cmd = [
            sys.executable,
            str(validation_script),
            "--databases", str(largest_db),
            "--mode", "canonical",
            "--output-json", str(output_json),
            "--progress-every", "100",
        ]

        try:
            print(f"[ParityValidation] Running validation on {largest_db.name}...")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)

            if process.returncode != 0:
                print(f"[ParityValidation] Validation failed: {stderr.decode()[:200]}")
                return results

            # Parse results
            if output_json.exists():
                with open(output_json) as f:
                    validation_results = json.load(f)

                results["total_games_checked"] = validation_results.get("total_games_checked", 0)
                failed = (
                    validation_results.get("games_with_semantic_divergence", 0) +
                    validation_results.get("games_with_structural_issues", 0)
                )
                results["games_passed"] = results["total_games_checked"] - failed
                results["games_failed"] = failed
                results["passed"] = validation_results.get("passed_canonical_parity_gate", False)

                if results["total_games_checked"] > 0:
                    results["failure_rate"] = results["games_failed"] / results["total_games_checked"]

            # Update state
            self.state.last_parity_validation = time.time()
            self.state.parity_validation_passed = results["passed"]
            self.state.parity_games_passed = results["games_passed"]
            self.state.parity_games_failed = results["games_failed"]

            if results["passed"]:
                print(f"[ParityValidation] PASSED: {results['games_passed']}/{results['total_games_checked']} games")
            else:
                print(f"[ParityValidation] FAILED: {results['games_failed']} games with issues "
                      f"(failure rate: {results['failure_rate']:.1%})")

        except asyncio.TimeoutError:
            print("[ParityValidation] Validation timed out")
        except Exception as e:
            print(f"[ParityValidation] Error: {e}")

        return results

    def should_run_parity_validation(self, min_interval_seconds: int = 3600) -> bool:
        """Check if parity validation should run."""
        now = time.time()
        return now - self.state.last_parity_validation >= min_interval_seconds

    # =========================================================================
    # CMA-ES Hyperparameter Optimization
    # =========================================================================

    async def run_cmaes_optimization(
        self,
        config_key: Optional[str] = None,
        reason: str = "manual",
    ) -> Dict[str, Any]:
        """Run CMA-ES hyperparameter optimization.

        This is typically triggered by:
        - Feedback controller detecting plateau
        - Manual request
        - Consecutive promotion failures
        """
        if self.state.cmaes_in_progress:
            print("[CMA-ES] Optimization already in progress")
            return {"success": False, "reason": "already_running"}

        print(f"[CMA-ES] === Starting CMA-ES Optimization === (reason: {reason})")

        # Find CMA-ES script
        cmaes_script = AI_SERVICE_ROOT / "scripts" / "cmaes_optimize.py"
        if not cmaes_script.exists():
            cmaes_script = AI_SERVICE_ROOT / "scripts" / "run_cmaes_weights.py"

        if not cmaes_script.exists():
            print("[CMA-ES] No CMA-ES script found")
            return {"success": False, "reason": "script_not_found"}

        results = {
            "success": False,
            "reason": reason,
            "best_weights": None,
            "best_fitness": None,
        }

        run_id = f"cmaes_{int(time.time())}"
        self.state.cmaes_in_progress = True
        self.state.cmaes_run_id = run_id
        self.state.cmaes_started_at = time.time()
        self._save_state()

        try:
            # Build command
            cmd = [
                sys.executable,
                str(cmaes_script),
                "--population-size", "10",
                "--generations", "20",
                "--output", str(AI_SERVICE_ROOT / "data" / f"cmaes_weights_{run_id}.json"),
            ]

            if config_key:
                parts = config_key.rsplit("_", 1)
                board_type = parts[0]
                num_players = int(parts[1].replace("p", ""))
                cmd.extend(["--board-type", board_type, "--num-players", str(num_players)])

            print(f"[CMA-ES] Running: {' '.join(cmd)}")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )

            # Use a generous timeout for CMA-ES
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=7200)

            if process.returncode == 0:
                results["success"] = True
                print(f"[CMA-ES] Optimization completed successfully")

                # Load best weights
                weights_file = AI_SERVICE_ROOT / "data" / f"cmaes_weights_{run_id}.json"
                if weights_file.exists():
                    with open(weights_file) as f:
                        data = json.load(f)
                    results["best_weights"] = data.get("best_weights")
                    results["best_fitness"] = data.get("best_fitness")
                    self.state.cmaes_best_weights = results["best_weights"]
                    print(f"[CMA-ES] Best fitness: {results['best_fitness']}")
            else:
                print(f"[CMA-ES] Optimization failed: {stderr.decode()[:200]}")
                results["reason"] = stderr.decode()[:200]

        except asyncio.TimeoutError:
            print("[CMA-ES] Optimization timed out")
            results["reason"] = "timeout"
        except Exception as e:
            print(f"[CMA-ES] Error: {e}")
            results["reason"] = str(e)
        finally:
            self.state.cmaes_in_progress = False
            self._save_state()

        # Publish completion event
        await self.event_bus.publish(DataEvent(
            event_type=DataEventType.CMAES_COMPLETED if results["success"] else DataEventType.TRAINING_COMPLETED,
            payload=results,
        ))

        return results

    async def _handle_cmaes_trigger(self, event: DataEvent):
        """Handle CMA-ES trigger events from feedback controller."""
        payload = event.payload
        reason = payload.get("reason", "feedback_trigger")
        configs = payload.get("configs", [])
        config_key = configs[0] if configs else None

        print(f"[CMA-ES] Trigger received: {reason}")
        await self.run_cmaes_optimization(config_key=config_key, reason=reason)

    # =========================================================================
    # Failure Event Handlers
    # =========================================================================

    async def _on_training_failed(self, event: DataEvent):
        """Handle training failure events - log, update health, consider retry."""
        payload = event.payload
        config_key = payload.get('config', 'unknown')
        error = payload.get('error', 'unknown error')
        duration = payload.get('duration', 0)

        print(f"[FailureHandler] TRAINING_FAILED for {config_key}: {error}")

        # Update health tracker
        if self.health_tracker:
            self.health_tracker.record_failure("training_scheduler", f"{config_key}: {str(error)[:200]}")

        # Notify feedback controller if available
        if self.feedback:
            try:
                if hasattr(self.feedback, 'on_stage_failed'):
                    await self.feedback.on_stage_failed('training', {
                        'config_key': config_key,
                        'error': str(error),
                        'duration': duration,
                    })
            except Exception as e:
                print(f"[FailureHandler] Error notifying feedback controller: {e}")

        # Check if we should increase data collection (might help with training issues)
        if self.feedback:
            pending = self.feedback.get_pending_actions()
            for action in pending:
                if action.action == FeedbackAction.INCREASE_DATA_COLLECTION:
                    await self._adjust_selfplay_rate(action.params.get('multiplier', 1.2), action.reason)

    async def _on_evaluation_failed(self, event: DataEvent):
        """Handle evaluation failure events - log, update health, skip model."""
        payload = event.payload
        config_key = payload.get('config', 'unknown')
        model_id = payload.get('model_id', 'unknown')
        error = payload.get('error', 'unknown error')

        print(f"[FailureHandler] EVALUATION_FAILED for {config_key}/{model_id}: {error}")

        # Update health tracker
        if self.health_tracker:
            self.health_tracker.record_failure("evaluator", f"{config_key}/{model_id}: {str(error)[:200]}")

        # Notify feedback controller
        if self.feedback:
            try:
                if hasattr(self.feedback, 'on_stage_failed'):
                    await self.feedback.on_stage_failed('evaluation', {
                        'config_key': config_key,
                        'model_id': model_id,
                        'error': str(error),
                    })
            except Exception as e:
                print(f"[FailureHandler] Error notifying feedback controller: {e}")

    async def _on_promotion_failed(self, event: DataEvent):
        """Handle promotion failure events - log and record."""
        payload = event.payload
        config_key = payload.get('config', 'unknown')
        model_id = payload.get('model_id', 'unknown')
        error = payload.get('error', 'unknown error')

        print(f"[FailureHandler] PROMOTION_FAILED for {config_key}/{model_id}: {error}")

        # Update health tracker
        if self.health_tracker:
            self.health_tracker.record_failure("promoter", f"{config_key}/{model_id}: {str(error)[:200]}")

    async def _on_data_sync_failed(self, event: DataEvent):
        """Handle data sync failure events - log and potentially adjust sync strategy."""
        payload = event.payload
        host = payload.get('host', 'unknown')
        error = payload.get('error', 'unknown error')
        retry_count = payload.get('retry_count', 0)

        print(f"[FailureHandler] DATA_SYNC_FAILED for host {host} (retry={retry_count}): {error}")

        # Update health tracker - use data_collector as closest match
        if self.health_tracker:
            self.health_tracker.record_failure("data_collector", f"sync:{host}: {str(error)[:200]}")

        # If circuit breaker is available, it will automatically handle backoff
        # Just log for awareness here
        if retry_count >= 3:
            print(f"[FailureHandler] Multiple sync failures for {host} - circuit breaker may open")

    # =========================================================================
    # FeedbackSignalRouter Handlers
    # =========================================================================

    async def _handle_increase_data_collection(self, signal: 'FeedbackSignal') -> bool:
        """Handle INCREASE_DATA_COLLECTION signals - boost selfplay games."""
        try:
            print(f"[FeedbackRouter] INCREASE_DATA_COLLECTION: {signal.reason} (magnitude={signal.magnitude:.2f})")

            # Increase data collection rate through adaptive controller if available
            if hasattr(self, 'training_scheduler') and self.training_scheduler:
                # Scale up selfplay games temporarily
                current_games = self.config.training.games_per_config
                boost_factor = 1.0 + (signal.magnitude * 0.5)  # Up to 50% boost
                boosted_games = int(current_games * boost_factor)

                print(f"[FeedbackRouter] Boosting games_per_config: {current_games} -> {boosted_games}")
                # Store original for later restoration
                if not hasattr(self, '_original_games_per_config'):
                    self._original_games_per_config = current_games
                self.config.training.games_per_config = boosted_games

            return True
        except Exception as e:
            print(f"[FeedbackRouter] Error handling INCREASE_DATA_COLLECTION: {e}")
            return False

    async def _handle_urgent_retraining(self, signal: 'FeedbackSignal') -> bool:
        """Handle URGENT_RETRAINING signals - fast-track training for regression recovery."""
        try:
            print(f"[FeedbackRouter] URGENT_RETRAINING: {signal.reason} (magnitude={signal.magnitude:.2f})")

            # Fast-track training by reducing trigger threshold temporarily
            if hasattr(self, 'training_scheduler') and self.training_scheduler:
                # Allow training with fewer games for urgent recovery
                original_threshold = self.config.training.trigger_threshold_games
                urgent_threshold = max(100, original_threshold // 2)

                print(f"[FeedbackRouter] Urgent retrain: reducing threshold {original_threshold} -> {urgent_threshold}")
                if not hasattr(self, '_original_threshold'):
                    self._original_threshold = original_threshold
                self.config.training.trigger_threshold_games = urgent_threshold

            return True
        except Exception as e:
            print(f"[FeedbackRouter] Error handling URGENT_RETRAINING: {e}")
            return False

    async def _handle_cmaes_feedback_signal(self, signal: 'FeedbackSignal') -> bool:
        """Handle TRIGGER_CMAES signals - initiate hyperparameter optimization."""
        try:
            print(f"[FeedbackRouter] TRIGGER_CMAES: {signal.reason} (magnitude={signal.magnitude:.2f})")

            # Delegate to PBT integration if available (CMA-ES is a type of hyperparameter optimization)
            if hasattr(self, 'pbt_integration') and self.pbt_integration and self.pbt_integration.config.enabled:
                # The signal's target_stage typically indicates which config to optimize
                config_key = signal.target_stage
                print(f"[FeedbackRouter] Triggering CMA-ES optimization for {config_key}")

                # Emit CMAES_TRIGGERED event for the pipeline to handle
                if HAS_DATA_EVENTS:
                    try:
                        event = DataEvent(
                            event_type=DataEventType.CMAES_TRIGGERED,
                            payload={
                                'config': config_key,
                                'reason': signal.reason,
                                'magnitude': signal.magnitude,
                            },
                            source='feedback_router'
                        )
                        await self.event_bus.publish(event)
                    except Exception as e:
                        print(f"[FeedbackRouter] Failed to emit CMAES_TRIGGERED: {e}")

            return True
        except Exception as e:
            print(f"[FeedbackRouter] Error handling TRIGGER_CMAES: {e}")
            return False

    async def _handle_scale_up_selfplay(self, signal: 'FeedbackSignal') -> bool:
        """Handle SCALE_UP_SELFPLAY signals - increase concurrent selfplay for underutilized resources."""
        try:
            print(f"[FeedbackRouter] SCALE_UP_SELFPLAY: {signal.reason} (magnitude={signal.magnitude:.2f})")

            # Scale up concurrent selfplay workers
            if hasattr(self, 'data_collector') and self.data_collector:
                # Increase parallel selfplay capacity
                current_parallel = getattr(self.config.data_ingestion, 'parallel_hosts', 3)
                scale_factor = 1.0 + (signal.magnitude * 0.3)  # Up to 30% more parallel
                new_parallel = min(10, int(current_parallel * scale_factor))

                print(f"[FeedbackRouter] Scaling up parallel hosts: {current_parallel} -> {new_parallel}")
                self.config.data_ingestion.parallel_hosts = new_parallel

            # Try to start selfplay on idle hosts via SSH (fallback when P2P unavailable)
            await self._spawn_selfplay_on_idle_hosts()

            return True
        except Exception as e:
            print(f"[FeedbackRouter] Error handling SCALE_UP_SELFPLAY: {e}")
            return False

    # Host CPU capacities for utilization-based scaling
    HOST_CPU_CAPACITY = {
        "gh200": 72,  # GH200 instances have 72 CPUs
        "lambda_2xh100": 48,
        "lambda_h100": 30,
        "lambda_a10": 30,
        "vast_5090": 512,
        "vast_4060": 384,
        "vast_3070": 24,
        "vast_2060": 22,
    }

    def _get_host_cpu_capacity(self, host_name: str) -> int:
        """Get CPU capacity for a host based on name pattern."""
        name_lower = host_name.lower()
        for pattern, cpus in self.HOST_CPU_CAPACITY.items():
            if pattern in name_lower:
                return cpus
        return 16  # Default for unknown hosts

    async def _spawn_selfplay_on_idle_hosts(self) -> int:
        """Spawn selfplay on underutilized hosts based on CPU capacity.

        Returns number of hosts where selfplay was started.
        """
        import asyncio

        # Get all selfplay-capable hosts (GH200 + Lambda)
        selfplay_hosts = []
        if hasattr(self, 'data_collector') and self.data_collector:
            for host_state in self.data_collector.state.hosts.values():
                name_lower = host_state.name.lower()
                # Include GH200 and Lambda hosts (but not AWS which have low RAM)
                if any(x in name_lower for x in ['gh200', 'lambda_2xh100', 'lambda_a10']):
                    selfplay_hosts.append((
                        host_state.name,
                        f"{host_state.ssh_user}@{host_state.ssh_host}",
                        self._get_host_cpu_capacity(host_state.name)
                    ))

        if not selfplay_hosts:
            return 0

        underutilized_hosts = []

        # Check each host for selfplay processes
        async def check_host(name: str, ssh_target: str, cpus: int) -> tuple[str, str, int, int]:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", ssh_target,
                    "ps aux | grep -E 'selfplay|hybrid' | grep -v grep | wc -l",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                count = int(stdout.decode().strip() or "0")
                return (name, ssh_target, count, cpus)
            except Exception:
                return (name, ssh_target, -1, cpus)  # -1 means unreachable

        # Check all hosts in parallel
        results = await asyncio.gather(*[check_host(n, s, c) for n, s, c in selfplay_hosts])

        for name, ssh_target, count, cpus in results:
            if count == -1:
                print(f"[Utilization] {name}: unreachable")
            elif count < cpus * 0.5:  # Underutilized if < 50% capacity
                # Calculate how many more workers to start (target 70% utilization)
                target = int(cpus * 0.7)
                workers_needed = max(2, min(10, target - count))  # Start 2-10 workers
                underutilized_hosts.append((name, ssh_target, count, cpus, workers_needed))

        if not underutilized_hosts:
            print("[Utilization] All hosts adequately utilized")
            return 0

        print(f"[Utilization] Found {len(underutilized_hosts)} underutilized hosts:")
        for name, _, count, cpus, needed in underutilized_hosts:
            print(f"  {name}: {count}/{cpus} CPUs ({count/cpus*100:.0f}%), need +{needed}")

        # Start selfplay on underutilized hosts (process up to 6 hosts per cycle)
        # Use NN-based modes for higher quality training data
        nn_modes = ["nn-only", "best-vs-pool", "nn-only", "mcts-only", "descent-only"]
        started = 0
        for name, ssh_target, count, cpus, workers_needed in underutilized_hosts[:6]:
            try:
                # Start multiple workers based on capacity
                # Cycle through NN-based modes for varied training data
                mode_list = " ".join([nn_modes[i % len(nn_modes)] for i in range(workers_needed)])
                cmd = f'''cd ~/ringrift/ai-service && source venv/bin/activate && \\
                    mkdir -p data/selfplay/auto_$(date +%s) && \\
                    modes=({mode_list}) && \\
                    for i in $(seq 1 {workers_needed}); do \\
                        mode=${{modes[$((i-1))]}};\\
                        nohup python scripts/run_hybrid_selfplay.py \\
                            --board-type square8 --num-players 2 --num-games 50000 \\
                            --record-db data/games/selfplay.db \\
                            --output-dir data/selfplay/auto_$(date +%s)/$i \\
                            --engine-mode $mode --seed $((RANDOM + $i * 1000)) \\
                            > logs/auto_selfplay_$i.log 2>&1 & \\
                    done && echo "Started {workers_needed} selfplay workers"'''

                proc = await asyncio.create_subprocess_exec(
                    "ssh", "-o", "ConnectTimeout=10", ssh_target, cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await asyncio.wait_for(proc.communicate(), timeout=30)

                if proc.returncode == 0:
                    print(f"[Utilization] Started {workers_needed} selfplay workers on {name}")
                    started += workers_needed
            except Exception as e:
                print(f"[Utilization] Failed to start selfplay on {name}: {e}")

        return started

    async def _periodic_health_recovery(self):
        """Automatic health recovery - runs periodically to fix common issues.

        Checks and fixes:
        1. Idle hosts without selfplay
        2. Stale rsync temp files
        3. Stuck training processes
        4. Database lock cleanup
        """
        import asyncio
        import glob

        print("[HealthRecovery] Running periodic health check and recovery...")
        issues_fixed = 0

        # 1. Check for idle hosts and start selfplay
        started = await self._spawn_selfplay_on_idle_hosts()
        if started > 0:
            print(f"[HealthRecovery] Started selfplay on {started} idle hosts")
            issues_fixed += started

        # 2. Clean up stale rsync temp files (older than 1 hour)
        try:
            synced_dir = AI_SERVICE_ROOT / "data" / "games" / "synced"
            if synced_dir.exists():
                now = time.time()
                for host_dir in synced_dir.iterdir():
                    if host_dir.is_dir():
                        for temp_file in host_dir.glob(".*.db.*"):
                            file_age = now - temp_file.stat().st_mtime
                            if file_age > 3600:  # Older than 1 hour
                                temp_file.unlink()
                                print(f"[HealthRecovery] Removed stale temp file: {temp_file.name}")
                                issues_fixed += 1
        except Exception as e:
            print(f"[HealthRecovery] Error cleaning temp files: {e}")

        # 3. Check for stuck training (running > 4 hours without progress)
        if hasattr(self, 'training_scheduler') and self.training_scheduler:
            ts = self.training_scheduler
            if ts._training_process and ts.state.training_in_progress:
                training_duration = time.time() - ts.state.training_started_at
                if training_duration > 14400:  # 4 hours
                    print(f"[HealthRecovery] Training stuck for {training_duration/3600:.1f}h, killing...")
                    try:
                        ts._training_process.kill()
                        ts._training_process = None
                        ts.state.training_in_progress = False
                        ts._release_training_lock()
                        issues_fixed += 1
                    except Exception as e:
                        print(f"[HealthRecovery] Failed to kill stuck training: {e}")

        # 4. Check database lock timeouts
        try:
            coordination_db = AI_SERVICE_ROOT / "data" / "coordination.db"
            if coordination_db.exists():
                import sqlite3
                conn = sqlite3.connect(str(coordination_db), timeout=5)
                cursor = conn.cursor()
                # Clear stale locks older than 2 hours
                cursor.execute("""
                    DELETE FROM locks
                    WHERE acquired_at < datetime('now', '-2 hours')
                """)
                deleted = cursor.rowcount
                conn.commit()
                conn.close()
                if deleted > 0:
                    print(f"[HealthRecovery] Cleared {deleted} stale database locks")
                    issues_fixed += deleted
        except Exception:
            pass  # Non-critical

        # 5. Clean up stale processes on GPU hosts (tournaments/selfplay shouldn't run on expensive GPUs)
        # GPU hosts should focus on GPU training, not CPU-bound tournaments
        GPU_HOSTS = [
            ("lambda_h100", "ubuntu@209.20.157.81"),
            ("lambda_2xh100", "ubuntu@192.222.53.22"),
            ("lambda_a10", "ubuntu@150.136.65.197"),
        ]
        STALE_PROCESS_PATTERNS = [
            ("run_model_elo_tournament", 7200),  # 2 hours - tournaments should be quick
            ("tune_hyperparameters", 14400),     # 4 hours - HP tuning can be long but not forever
        ]

        for host_name, ssh_target in GPU_HOSTS:
            try:
                for pattern, max_seconds in STALE_PROCESS_PATTERNS:
                    # Find processes matching pattern older than threshold
                    check_cmd = f"ps -eo pid,etimes,args | grep '{pattern}' | grep -v grep | awk '$2 > {max_seconds} {{print $1}}'"
                    proc = await asyncio.create_subprocess_exec(
                        "ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", ssh_target, check_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)

                    stale_pids = stdout.decode().strip().split('\n')
                    stale_pids = [p for p in stale_pids if p.strip().isdigit()]

                    if stale_pids:
                        # Kill stale processes
                        kill_cmd = f"kill -9 {' '.join(stale_pids)} 2>/dev/null"
                        proc = await asyncio.create_subprocess_exec(
                            "ssh", "-o", "ConnectTimeout=5", ssh_target, kill_cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        await asyncio.wait_for(proc.communicate(), timeout=10)
                        print(f"[HealthRecovery] Killed {len(stale_pids)} stale {pattern} on {host_name}")
                        issues_fixed += len(stale_pids)
            except Exception as e:
                print(f"[HealthRecovery] Error checking stale processes on {host_name}: {e}")

        if issues_fixed > 0:
            print(f"[HealthRecovery] Fixed {issues_fixed} issues")
        else:
            print("[HealthRecovery] All systems healthy")

        return issues_fixed

    async def _health_recovery_loop(self):
        """Background loop that runs periodic health recovery every 5 minutes."""
        recovery_interval = 300  # 5 minutes

        print("[HealthRecovery] Recovery loop started (interval=5min)")

        while self._running:
            try:
                await self._periodic_health_recovery()
            except Exception as e:
                print(f"[HealthRecovery] Error in recovery loop: {e}")

            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=recovery_interval)
                break
            except asyncio.TimeoutError:
                pass

        print("[HealthRecovery] Recovery loop stopped")

    async def _handle_scale_down_selfplay(self, signal: 'FeedbackSignal') -> bool:
        """Handle SCALE_DOWN_SELFPLAY signals - reduce concurrent selfplay to avoid resource contention."""
        try:
            print(f"[FeedbackRouter] SCALE_DOWN_SELFPLAY: {signal.reason} (magnitude={signal.magnitude:.2f})")

            # Scale down concurrent selfplay workers
            if hasattr(self, 'data_collector') and self.data_collector:
                current_parallel = getattr(self.config.data_ingestion, 'parallel_hosts', 3)
                scale_factor = 1.0 - (signal.magnitude * 0.3)  # Down to 30% fewer parallel
                new_parallel = max(1, int(current_parallel * scale_factor))

                print(f"[FeedbackRouter] Scaling down parallel hosts: {current_parallel} -> {new_parallel}")
                self.config.data_ingestion.parallel_hosts = new_parallel

            return True
        except Exception as e:
            print(f"[FeedbackRouter] Error handling SCALE_DOWN_SELFPLAY: {e}")
            return False

    async def route_feedback_signal(self, signal: 'FeedbackSignal') -> List[Tuple[str, bool]]:
        """Route a feedback signal through the router.

        This can be called by components to route signals through the unified router.
        """
        if self.feedback_router is None:
            print(f"[FeedbackRouter] No router available, signal dropped: {signal.action.value}")
            return []
        return await self.feedback_router.route(signal)

    # =========================================================================
    # Diverse Tournament Distribution (Elo Calibration)
    # =========================================================================

    async def run_diverse_tournaments(
        self,
        games_per_config: int = 50,
        board_types: Optional[List[str]] = None,
        player_counts: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Run diverse tournaments across all board/player configurations.

        This provides richer Elo calibration by testing all AI types:
        - Random, heuristic, minimax, MCTS, neural
        - Across all board types and player counts

        Args:
            games_per_config: Games per board/player configuration
            board_types: Board types to include (default: all)
            player_counts: Player counts to include (default: all)

        Returns:
            Dictionary with tournament results summary
        """
        if not HAS_DIVERSE_TOURNAMENTS:
            print("[DiverseTournament] Module not available, skipping")
            return {"skipped": True, "reason": "module_not_available"}

        print("[DiverseTournament] === Starting Diverse Tournament Phase ===")

        # Default configuration
        board_types = board_types or ["square8", "square19", "hexagonal"]
        player_counts = player_counts or [2, 3, 4]
        iteration = self.state.total_evaluations + 1
        output_base = str(AI_SERVICE_ROOT / "data" / "tournaments" / f"iter{iteration}")

        try:
            # Build tournament configs
            configs = build_tournament_configs(
                board_types=board_types,
                player_counts=player_counts,
                games_per_config=games_per_config,
                output_base=output_base,
                seed=int(time.time()),
            )

            print(f"[DiverseTournament] Configurations: {len(configs)}")
            for config in configs:
                print(f"  {config.board_type} {config.num_players}p: {config.num_games} games")

            # Run locally (sequential) - for distributed, use p2p_orchestrator
            results = run_tournament_round_local(configs)

            # Aggregate results
            total_games = sum(r.games_completed for r in results)
            total_samples = sum(r.samples_generated for r in results)
            successful = sum(1 for r in results if r.success)

            summary = {
                "iteration": iteration,
                "configurations": len(configs),
                "successful": successful,
                "total_games": total_games,
                "total_samples": total_samples,
                "results": [
                    {
                        "config": f"{r.config.board_type}_{r.config.num_players}p",
                        "games": r.games_completed,
                        "samples": r.samples_generated,
                        "success": r.success,
                    }
                    for r in results
                ],
            }

            print(f"[DiverseTournament] Complete: {successful}/{len(configs)} configs, {total_games} games")

            # Publish event
            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.EVALUATION_COMPLETED,
                payload={
                    "type": "diverse_tournament",
                    **summary,
                }
            ))

            return summary

        except Exception as e:
            print(f"[DiverseTournament] Error: {e}")
            return {"success": False, "error": str(e)}

    def should_run_diverse_tournaments(self, min_interval_seconds: int = 21600) -> bool:
        """Check if diverse tournaments should run (default: every 6 hours)."""
        now = time.time()
        return now - self._last_diverse_tournament >= min_interval_seconds

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
                # Quick connectivity check with short timeout before starting background tasks
                import aiohttp
                base_url = self.p2p.config.p2p_base_url
                try:
                    timeout = aiohttp.ClientTimeout(total=3)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(f"{base_url}/health") as resp:
                            if resp.status != 200:
                                print(f"[P2P] P2P server at {base_url} not healthy (status={resp.status}), skipping integration")
                                return
                except Exception as conn_err:
                    print(f"[P2P] P2P server at {base_url} not reachable ({type(conn_err).__name__}), skipping integration")
                    return

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
        """Main data collection loop - runs every 60 seconds.

        When use_external_sync=True, this loop only monitors for new data
        synced by the external unified_data_sync.py service.
        """
        use_external = self.config.data_ingestion.use_external_sync
        if use_external:
            print("[DataCollection] External sync mode - monitoring synced directory...", flush=True)
        else:
            print("[DataCollection] Loop starting...", flush=True)
        _quality_check_counter = 0
        _quality_check_interval = 10  # Check quality every 10 sync cycles
        _last_game_count = 0  # For external sync mode monitoring

        while self._running:
            try:
                if use_external:
                    # External sync mode: just count games in synced directory
                    synced_dir = AI_SERVICE_ROOT / "data" / "games" / "synced"
                    new_games = 0
                    current_count = 0
                    if synced_dir.exists():
                        for db_path in synced_dir.rglob("*.db"):
                            if "schema" in db_path.name:
                                continue
                            try:
                                import sqlite3
                                conn = sqlite3.connect(db_path)
                                count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
                                conn.close()
                                current_count += count
                            except Exception:
                                pass
                        new_games = max(0, current_count - _last_game_count)
                        _last_game_count = current_count
                    self.health_tracker.record_success("data_collector")
                    if new_games > 0:
                        print(f"[DataCollection] External sync detected {new_games} new games (total: {current_count})")
                        # Update per-config game counts for training scheduler
                        self.data_collector._update_per_config_game_counts(new_games)
                else:
                    # Internal sync mode: actively pull from hosts
                    new_games = await self.data_collector.run_collection_cycle()
                    self.health_tracker.record_success("data_collector")
                    if new_games > 0:
                        print(f"[DataCollection] Synced {new_games} new games")

                # Check if training threshold reached (works in both modes)
                if new_games > 0:
                    trigger_config = self.training_scheduler.should_trigger_training()
                    if trigger_config:
                        await self.event_bus.publish(DataEvent(
                            event_type=DataEventType.TRAINING_THRESHOLD_REACHED,
                            payload={"config": trigger_config}
                        ))

                # Periodic data quality check (every 10 cycles, ~10 minutes)
                # Only run quality stats when using internal collector (has compute_quality_stats)
                _quality_check_counter += 1
                if _quality_check_counter >= _quality_check_interval:
                    _quality_check_counter = 0
                    if self.feedback and not use_external:
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
        """Main evaluation loop - shadow every 15 min, full every 1 hour.

        OPTIMIZED: Now runs shadow tournaments in parallel (up to 3 concurrent)
        instead of one-at-a-time, reducing evaluation cycle time by 3-6x.
        """
        # Configuration for parallel evaluation
        max_concurrent_shadow = 3  # Run up to 3 shadow tournaments in parallel

        while self._running:
            try:
                now = time.time()

                # Get promotion velocity for adaptive interval calculation
                promotion_velocity = 0.0
                if hasattr(self, 'training_scheduler') and hasattr(self.training_scheduler, '_promotion_history'):
                    recent_promo = [t for t in self.training_scheduler._promotion_history if now - t < 3600]
                    promotion_velocity = len(recent_promo)  # promotions per hour

                # Get adaptive interval (adjusts based on success rate, duration, and promotion velocity)
                adaptive_interval = self.shadow_tournament.get_adaptive_interval(promotion_velocity)

                # Collect all configs that need shadow evaluation
                configs_needing_eval = []
                for config_key in self.state.configs:
                    last_eval = self._last_shadow_eval.get(config_key, 0)
                    if now - last_eval >= adaptive_interval:
                        configs_needing_eval.append(config_key)

                # Sort by priority (highest priority = most urgent to evaluate)
                # This ensures underperforming configs get evaluated first
                if configs_needing_eval:
                    prioritized = self.config_priority.get_prioritized_configs(
                        {k: self.state.configs[k] for k in configs_needing_eval}
                    )
                    configs_needing_eval = [k for k, _ in prioritized]

                    # Log priority info for top configs
                    if prioritized and len(prioritized) > 1:
                        top_priority = prioritized[0]
                        print(f"[Evaluation] Highest priority: {top_priority[0]} (score={top_priority[1]:.1f})")

                # Run shadow tournaments in parallel (major optimization!)
                if configs_needing_eval:
                    print(f"[Evaluation] {len(configs_needing_eval)} configs need shadow evaluation")
                    results = await self.shadow_tournament.run_parallel_shadow_tournaments(
                        configs_needing_eval,
                        max_concurrent=max_concurrent_shadow
                    )

                    # Update tracking for successful evaluations
                    for result in results:
                        config_key = result.get("config")
                        if config_key and result.get("success", False):
                            self._last_shadow_eval[config_key] = now
                            self.health_tracker.record_success("evaluator")

                # Check for full tournament
                if now - self._last_full_eval >= self.config.evaluation.full_tournament_interval_seconds:
                    print("[Evaluation] Running full tournament")
                    await self.shadow_tournament.run_full_tournament()
                    self._last_full_eval = now
                    self.health_tracker.record_success("evaluator")

                # Check for diverse tournaments (Elo calibration) - every 6 hours
                if self.should_run_diverse_tournaments(min_interval_seconds=21600):
                    print("[Evaluation] Running diverse tournaments for Elo calibration")
                    await self.run_diverse_tournaments(
                        games_per_config=self.config.evaluation.full_tournament_games
                    )
                    self._last_diverse_tournament = now
                    self.health_tracker.record_success("evaluator")

                # Check for model pruning (when model count exceeds threshold)
                if await self.model_pruning.should_run():
                    pruning_status = self.model_pruning.get_status()
                    print(f"[Evaluation] Model pruning triggered: {pruning_status['model_count']} models")
                    result = await self.model_pruning.run_pruning_cycle()
                    if result:
                        self.health_tracker.record_success("model_pruning")
                    else:
                        self.health_tracker.record_failure("model_pruning", "Pruning cycle failed")

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
        print("[TrainingLoop] Starting training loop", flush=True)
        while self._running:
            try:
                # Check if training completed
                result = await self.training_scheduler.check_training_status()
                if result:
                    print(f"[Training] Completed: {result}")
                    self.health_tracker.record_success("training_scheduler")

                    # Run tier gating checks after training completes
                    promotions = await self.run_tier_gating_checks()
                    if promotions:
                        print(f"[Training] Tier promotions after training: {promotions}")

                # Check if we should start training
                if not self.state.training_in_progress:
                    trigger_config = self.training_scheduler.should_trigger_training()
                    print(f"[TrainingLoop] Check: trigger_config={trigger_config}", flush=True)
                    if trigger_config:
                        # Run parity validation gate before training
                        if self.should_run_parity_validation(min_interval_seconds=1800):
                            parts = trigger_config.rsplit("_", 1)
                            board_type = parts[0]
                            num_players = int(parts[1].replace("p", ""))
                            parity_result = await self.run_parity_validation(
                                board_type=board_type, num_players=num_players
                            )
                            if not parity_result.get("passed", True):
                                failure_rate = parity_result.get("failure_rate", 0)
                                max_rate = self.config.feedback.max_parity_failure_rate if hasattr(self.config, 'feedback') else 0.10
                                if failure_rate > max_rate:
                                    print(f"[Training] BLOCKED by parity validation: {failure_rate:.1%} > {max_rate:.1%}")
                                    continue

                            # Record data quality in improvement optimizer for acceleration
                            if HAS_IMPROVEMENT_OPTIMIZER:
                                try:
                                    optimizer = get_improvement_optimizer()
                                    parity_success = 1.0 - parity_result.get("failure_rate", 0)
                                    data_quality = self.feedback.state.data_quality_score if self.feedback else 1.0
                                    rec = optimizer.record_data_quality(parity_success, data_quality)
                                    if rec.signal in (ImprovementSignal.QUALITY_DATA_SURGE, ImprovementSignal.DATA_QUALITY_HIGH):
                                        print(f"[ImprovementOptimizer] High data quality detected: "
                                              f"parity={parity_success:.1%}, quality={data_quality:.2f}")
                                except Exception:
                                    pass  # Don't block training for optimizer errors

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
                    success = await self.model_promoter.execute_promotion(candidate)

                    # Record successful promotion for dynamic threshold calculation
                    if success:
                        self.training_scheduler.record_promotion()
                        # Update config state
                        config_key = candidate.get("config")
                        if config_key and config_key in self.state.configs:
                            self.state.configs[config_key].last_promotion_time = time.time()

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

                # Periodic database integrity check (every 6th health check = ~30 min)
                if hasattr(self, "_db_integrity_check_counter"):
                    self._db_integrity_check_counter += 1
                else:
                    self._db_integrity_check_counter = 0

                if self._db_integrity_check_counter % 6 == 0:
                    try:
                        db_results = check_and_repair_databases(
                            data_dir=AI_SERVICE_ROOT / "data" / "games",
                            auto_repair=True,
                            log_prefix="[UnifiedLoop]"
                        )
                        if db_results["corrupted"] > 0:
                            print(f"[DBIntegrity] Checked {db_results['checked']} DBs: "
                                  f"{db_results['corrupted']} corrupted, "
                                  f"{db_results['recovered']} recovered, "
                                  f"{db_results['failed']} failed")
                        elif self.config.verbose:
                            print(f"[DBIntegrity] All {db_results['checked']} databases healthy")
                    except Exception as db_e:
                        print(f"[DBIntegrity] Error checking databases: {db_e}")

            except Exception as e:
                print(f"[HealthCheck] Error running health check: {e}")
                if HAS_PROMETHEUS:
                    LOOP_ERRORS_TOTAL.labels(loop="health_check", error_type=type(e).__name__).inc()

            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=health_check_interval)
                break
            except asyncio.TimeoutError:
                pass

    async def _utilization_optimization_loop(self):
        """Actively optimize cluster utilization toward 60-80% target.

        This loop coordinates with the resource optimizer to:
        1. Collect utilization metrics from all hosts
        2. Calculate optimal job distribution
        3. Emit scaling recommendations
        4. Provide feedback metrics for self-improvement
        """
        if not HAS_RESOURCE_OPTIMIZER or self.resource_optimizer is None:
            print("[Utilization] Resource optimizer not available - skipping")
            return

        optimization_interval = 30  # Check every 30 seconds
        last_recommendation_time = 0.0
        last_feedback_adjustment_time = 0.0
        recommendation_cooldown = 60.0  # Don't spam recommendations
        feedback_adjustment_interval = 300.0  # Apply rate feedback adjustment every 5 minutes
        consecutive_optimal = 0
        consecutive_suboptimal = 0

        print("[Utilization] Optimization loop started (target: 60-80%)")

        while self._running:
            try:
                # Collect utilization from backend workers
                if self.backend is not None:
                    try:
                        workers = await self.backend.get_available_workers()
                        for w in workers:
                            # Report to shared optimizer
                            resources = NodeResources(
                                node_id=w.name,
                                cpu_percent=w.cpu_percent,
                                memory_percent=w.memory_percent,
                                active_jobs=w.active_jobs,
                                has_gpu=bool(w.metadata.get("gpu")),
                                gpu_name=w.metadata.get("gpu", ""),
                            )
                            self.resource_optimizer.report_node_resources(resources)
                    except Exception as e:
                        if self.config.verbose:
                            print(f"[Utilization] Backend worker query error: {e}")

                # Get cluster state
                cluster_state = self.resource_optimizer.get_cluster_state()

                # Check if within target range (60-80%)
                cpu_in_range = 60 <= cluster_state.total_cpu_util <= 80
                gpu_in_range = cluster_state.gpu_node_count == 0 or 60 <= cluster_state.total_gpu_util <= 80

                if cpu_in_range and gpu_in_range:
                    consecutive_optimal += 1
                    consecutive_suboptimal = 0
                    if consecutive_optimal == 6 and self.config.verbose:  # ~3 min in optimal
                        print(f"[Utilization] Optimal: CPU={cluster_state.total_cpu_util:.1f}%, GPU={cluster_state.total_gpu_util:.1f}%")
                else:
                    consecutive_suboptimal += 1
                    consecutive_optimal = 0

                # Get optimization recommendation
                now = time.time()
                if consecutive_suboptimal >= 3 and now - last_recommendation_time >= recommendation_cooldown:
                    rec = self.resource_optimizer.get_optimization_recommendation()

                    if rec.action != ScaleAction.NONE:
                        print(
                            f"[Utilization] {rec.action.value}: {rec.reason} "
                            f"(confidence={rec.confidence:.2f}, adjustment={rec.adjustment})"
                        )

                        # Apply recommendation via P2P if available
                        if self.p2p is not None and rec.action == ScaleAction.SCALE_UP:
                            try:
                                # Request P2P to scale selfplay
                                await self._request_p2p_scale(
                                    direction="up",
                                    magnitude=abs(rec.adjustment),
                                    reason=rec.reason,
                                )
                            except Exception as e:
                                print(f"[Utilization] P2P scale request failed: {e}")

                        elif self.p2p is not None and rec.action == ScaleAction.SCALE_DOWN:
                            try:
                                await self._request_p2p_scale(
                                    direction="down",
                                    magnitude=abs(rec.adjustment),
                                    reason=rec.reason,
                                )
                            except Exception as e:
                                print(f"[Utilization] P2P scale request failed: {e}")

                        # Record the action
                        self.resource_optimizer.record_optimization_action(rec)
                        last_recommendation_time = now

                # Update feedback for self-improvement
                if self.feedback is not None and consecutive_suboptimal >= 6:
                    try:
                        # Trigger feedback on sustained suboptimal utilization
                        # This can influence selfplay rate and training scheduling
                        await self.feedback.on_stage_complete('utilization', {
                            'cpu_util': cluster_state.total_cpu_util,
                            'gpu_util': cluster_state.total_gpu_util,
                            'in_target_range': cpu_in_range and gpu_in_range,
                            'node_count': cluster_state.cpu_node_count,
                            'total_jobs': cluster_state.total_jobs,
                        })
                    except Exception as e:
                        if self.config.verbose:
                            print(f"[Utilization] Feedback error: {e}")

                # Prometheus metrics
                if HAS_PROMETHEUS:
                    LOOP_CYCLES_TOTAL.labels(loop="utilization").inc()

                # Periodic feedback-based rate adjustment (every 5 minutes)
                # This automatically adjusts selfplay rate to maintain 60-80% utilization
                now = time.time()
                if now - last_feedback_adjustment_time >= feedback_adjustment_interval:
                    if apply_feedback_adjustment is not None:
                        try:
                            new_rate = apply_feedback_adjustment(requestor="unified_loop_utilization")
                            status = get_utilization_status() if get_utilization_status else {}
                            cpu_util = status.get('cpu_util', 0)
                            gpu_util = status.get('gpu_util', 0)
                            util_status_text = status.get('status', 'unknown')

                            # Update Prometheus metrics for cluster utilization tracking
                            if HAS_PROMETHEUS:
                                SELFPLAY_RATE.set(new_rate)
                                # Map status text to numeric: below=-1, optimal=0, above=1
                                status_value = 0 if util_status_text == 'optimal' else (-1 if 'below' in util_status_text else 1)
                                UTILIZATION_STATUS.set(status_value)
                                OPTIMIZER_IN_TARGET.labels(resource='cpu').set(1 if 60 <= cpu_util <= 80 else 0)
                                OPTIMIZER_IN_TARGET.labels(resource='gpu').set(1 if 60 <= gpu_util <= 80 else 0)

                            if self.config.verbose:
                                print(f"[Utilization] Feedback adjustment applied: rate={new_rate}/hr "
                                      f"(CPU={cpu_util:.1f}%, GPU={gpu_util:.1f}%, status={util_status_text})")
                            last_feedback_adjustment_time = now
                        except Exception as e:
                            if self.config.verbose:
                                print(f"[Utilization] Feedback adjustment error: {e}")

            except Exception as e:
                print(f"[Utilization] Error: {e}")
                if HAS_PROMETHEUS:
                    LOOP_ERRORS_TOTAL.labels(loop="utilization", error_type=type(e).__name__).inc()

            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=optimization_interval)
                break
            except asyncio.TimeoutError:
                pass

        print("[Utilization] Optimization loop stopped")

    async def _request_p2p_scale(self, direction: str, magnitude: int, reason: str) -> None:
        """Request P2P orchestrator to scale selfplay jobs.

        Args:
            direction: "up" or "down"
            magnitude: Number of jobs to add/remove
            reason: Reason for scaling
        """
        if self.p2p is None:
            return

        try:
            if direction == "up":
                # Request more selfplay capacity
                result = await self.p2p.request_scale_up(jobs=magnitude, reason=reason)
                if result.get("success"):
                    print(f"[Utilization] P2P scale-up accepted: +{magnitude} jobs")
            else:
                # Request reduced selfplay capacity
                result = await self.p2p.request_scale_down(jobs=magnitude, reason=reason)
                if result.get("success"):
                    print(f"[Utilization] P2P scale-down accepted: -{magnitude} jobs")
        except AttributeError:
            # P2P integration may not have these methods yet
            if self.config.verbose:
                print(f"[Utilization] P2P integration doesn't support scaling API")
        except Exception as e:
            print(f"[Utilization] P2P scale request error: {e}")

    async def get_backend_workers(self) -> List[Dict[str, Any]]:
        """Query available workers from the execution backend.

        Returns:
            List of worker status dictionaries with name, available, and metadata.
        """
        if self.backend is None:
            return []

        try:
            workers = await self.backend.get_available_workers()
            return [
                {
                    "name": w.name,
                    "available": w.available,
                    "cpu_percent": w.cpu_percent,
                    "memory_percent": w.memory_percent,
                    "active_jobs": w.active_jobs,
                    "last_seen": w.last_seen,
                    "metadata": w.metadata,
                }
                for w in workers
            ]
        except Exception as e:
            print(f"[Backend] Error querying workers: {e}")
            return []

    async def run_distributed_selfplay(
        self,
        games: int,
        board_type: str = "square8",
        num_players: int = 2,
        model_path: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Run selfplay games using the execution backend.

        Distributes games across available workers using the configured backend.

        Args:
            games: Total number of games to generate
            board_type: Board type for games
            num_players: Number of players per game
            model_path: Optional path to model weights

        Returns:
            List of job results with output and status
        """
        if self.backend is None:
            print("[Backend] No execution backend available - cannot run distributed selfplay")
            return []

        try:
            results = await self.backend.run_selfplay(
                games=games,
                board_type=board_type,
                num_players=num_players,
                model_path=model_path,
                **kwargs,
            )

            # Convert JobResult to dict
            return [
                {
                    "job_id": r.job_id,
                    "success": r.success,
                    "worker": r.worker,
                    "duration_seconds": r.duration_seconds,
                    "error": r.error,
                }
                for r in results
            ]
        except Exception as e:
            print(f"[Backend] Error running distributed selfplay: {e}")
            return []

    async def sync_backend_data(self) -> Dict[str, int]:
        """Sync game data from workers using the execution backend.

        Returns:
            Dict mapping worker names to number of games synced.
        """
        if self.backend is None:
            return {}

        try:
            return await self.backend.sync_data()
        except Exception as e:
            print(f"[Backend] Error syncing data: {e}")
            return {}

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

    async def _gauntlet_culling_loop(self):
        """Gauntlet evaluation and model culling loop.

        Periodically checks if any config has:
        1. Many unrated models → run gauntlet evaluation
        2. Model count > 100 → cull bottom 75% (keeping top quartile)

        Respects uncertainty: models with < 20 games are protected from culling.
        """
        if not HAS_GAUNTLET or self.gauntlet is None or self.culler is None:
            print("[Gauntlet] Not available - skipping")
            return

        # Initial delay to let other loops stabilize
        await asyncio.sleep(60)

        print("[Gauntlet] Gauntlet evaluation and model culling loop started")

        while self._running:
            try:
                now = time.time()

                # Check if enough time has passed since last check
                if now - self._last_gauntlet_check >= self._gauntlet_interval:
                    self._last_gauntlet_check = now

                    # Check each of the 9 configs
                    for config_key in GAUNTLET_CONFIG_KEYS:
                        # Check for unrated models
                        unrated_count = len(self.gauntlet.get_unrated_models(config_key))
                        model_count = self.culler.count_models(config_key)

                        # Run gauntlet if many unrated models
                        if unrated_count > 20:
                            print(f"[Gauntlet] {config_key}: {unrated_count} unrated models - running gauntlet")
                            try:
                                result = await self.gauntlet.run_gauntlet(config_key)
                                print(f"[Gauntlet] {config_key}: Evaluated {result.models_evaluated} models")
                            except Exception as e:
                                print(f"[Gauntlet] {config_key}: Gauntlet error: {e}")

                        # Cull if over threshold
                        if self.culler.needs_culling(config_key):
                            print(f"[Gauntlet] {config_key}: {model_count} models - culling to top quartile")
                            try:
                                cull_result = self.culler.check_and_cull(config_key)
                                print(
                                    f"[Gauntlet] {config_key}: Culled {cull_result.culled}, "
                                    f"kept {cull_result.kept}"
                                )
                            except Exception as e:
                                print(f"[Gauntlet] {config_key}: Culling error: {e}")

                        # Small delay between configs to avoid overload
                        await asyncio.sleep(1)

            except Exception as e:
                print(f"[Gauntlet] Error in gauntlet/culling loop: {e}")

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=300  # Check every 5 minutes (inner interval check handles 30 min)
                )
                break
            except asyncio.TimeoutError:
                pass

    async def _elo_sync_loop(self):
        """Elo database synchronization loop for cluster-wide consistency.

        Periodically syncs the unified_elo.db with other nodes in the cluster
        to ensure all nodes have consistent Elo ratings for model evaluation
        and promotion decisions.

        Uses multi-transport failover (Tailscale, SSH, HTTP, aria2) with
        circuit breakers for fault tolerance.
        """
        if not HAS_ELO_SYNC or self.elo_sync_manager is None:
            print("[EloSync] Not available - skipping")
            return

        # Initial delay to let other systems stabilize
        await asyncio.sleep(30)

        # Initialize the sync manager
        try:
            await self.elo_sync_manager.initialize()
            print(f"[EloSync] Initialized - {self.elo_sync_manager.state.local_match_count} local matches")
        except Exception as e:
            print(f"[EloSync] Initialization failed: {e}")
            return

        print(f"[EloSync] Cluster-wide Elo sync loop started (interval: {self._elo_sync_interval}s)")

        while self._running:
            try:
                now = time.time()

                # Check if enough time has passed since last sync
                if now - self._last_elo_sync >= self._elo_sync_interval:
                    self._last_elo_sync = now

                    # Sync with cluster
                    success = await self.elo_sync_manager.sync_with_cluster()

                    if success:
                        print(
                            f"[EloSync] Sync complete: {self.elo_sync_manager.state.local_match_count} matches, "
                            f"synced from {self.elo_sync_manager.state.synced_from}"
                        )
                    else:
                        errors = self.elo_sync_manager.state.sync_errors[-3:]
                        print(f"[EloSync] Sync failed - recent errors: {errors}")

            except Exception as e:
                print(f"[EloSync] Error in sync loop: {e}")

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=60  # Check shutdown every minute
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
                    # Data collection events
                    "new_games",
                    "sync_started",
                    "sync_completed",
                    "sync_failed",
                    # Training events
                    "training_threshold",
                    "training_started",
                    "training_progress",
                    "training_completed",
                    "training_failed",
                    # Evaluation events
                    "evaluation_started",
                    "evaluation_progress",
                    "evaluation_completed",
                    "evaluation_failed",
                    "elo_updated",
                    # Promotion events
                    "promotion_candidate",
                    "promotion_started",
                    "model_promoted",
                    "promotion_failed",
                    "promotion_rejected",
                    # Curriculum events
                    "curriculum_rebalanced",
                    "weight_updated",
                    "elo_significant_change",
                    # System events
                    "daemon_started",
                    "daemon_stopped",
                    "host_online",
                    "host_offline",
                    "error",
                    # Optimization events
                    "cmaes_triggered",
                    "nas_triggered",
                    "plateau_detected",
                    "hyperparameter_updated",
                ]
            )
            print(f"[CrossProcess] Subscribed as {subscriber_id}")
        except Exception as e:
            print(f"[CrossProcess] Failed to initialize: {e}")
            return

        # Mapping from cross-process event types to local DataEventType
        EVENT_TYPE_MAP = {
            # Data collection events
            "new_games": DataEventType.NEW_GAMES_AVAILABLE,
            "sync_started": DataEventType.DATA_SYNC_STARTED,
            "sync_completed": DataEventType.DATA_SYNC_COMPLETED,
            "sync_failed": DataEventType.DATA_SYNC_FAILED,
            # Training events
            "training_threshold": DataEventType.TRAINING_THRESHOLD_REACHED,
            "training_started": DataEventType.TRAINING_STARTED,
            "training_progress": DataEventType.TRAINING_PROGRESS,
            "training_completed": DataEventType.TRAINING_COMPLETED,
            "training_failed": DataEventType.TRAINING_FAILED,
            # Evaluation events
            "evaluation_started": DataEventType.EVALUATION_STARTED,
            "evaluation_progress": DataEventType.EVALUATION_PROGRESS,
            "evaluation_completed": DataEventType.EVALUATION_COMPLETED,
            "evaluation_failed": DataEventType.EVALUATION_FAILED,
            "elo_updated": DataEventType.ELO_UPDATED,
            # Promotion events
            "promotion_candidate": DataEventType.PROMOTION_CANDIDATE,
            "promotion_started": DataEventType.PROMOTION_STARTED,
            "model_promoted": DataEventType.MODEL_PROMOTED,
            "promotion_failed": DataEventType.PROMOTION_FAILED,
            "promotion_rejected": DataEventType.PROMOTION_REJECTED,
            # Curriculum events
            "curriculum_rebalanced": DataEventType.CURRICULUM_REBALANCED,
            "weight_updated": DataEventType.WEIGHT_UPDATED,
            "elo_significant_change": DataEventType.ELO_SIGNIFICANT_CHANGE,
            # System events
            "daemon_started": DataEventType.DAEMON_STARTED,
            "daemon_stopped": DataEventType.DAEMON_STOPPED,
            "host_online": DataEventType.HOST_ONLINE,
            "host_offline": DataEventType.HOST_OFFLINE,
            "error": DataEventType.ERROR,
            # Optimization events
            "cmaes_triggered": DataEventType.CMAES_TRIGGERED,
            "nas_triggered": DataEventType.NAS_TRIGGERED,
            "plateau_detected": DataEventType.PLATEAU_DETECTED,
            "hyperparameter_updated": DataEventType.HYPERPARAMETER_UPDATED,
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
            # Use return_exceptions=True to prevent one loop's failure from crashing the entire system
            loop_names = [
                "data_collection", "evaluation", "training", "promotion",
                "curriculum", "metrics", "health_check", "utilization_optimization",
                "hp_tuning_sync", "external_drive_sync", "pbt", "nas",
                "per", "cross_process_event", "health_recovery", "gauntlet_culling",
                "elo_sync"
            ]
            results = await asyncio.gather(
                self._data_collection_loop(),
                self._evaluation_loop(),
                self._training_loop(),
                self._promotion_loop(),
                self._curriculum_loop(),
                self._metrics_loop(),
                self._health_check_loop(),
                self._utilization_optimization_loop(),  # Active 60-80% targeting
                self._hp_tuning_sync_loop(),
                self._external_drive_sync_loop(),
                self._pbt_loop(),
                self._nas_loop(),
                self._per_loop(),
                self._cross_process_event_loop(),
                self._health_recovery_loop(),  # Automatic issue detection and healing
                self._gauntlet_culling_loop(),  # Model evaluation and top-quartile culling
                self._elo_sync_loop(),  # Cluster-wide Elo database consistency
                return_exceptions=True,  # Don't crash if one loop fails
            )
            # Log any exceptions from loops
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"[UnifiedLoop] ERROR in {loop_names[i]} loop: {result}")
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
