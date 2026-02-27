"""Unified Loop Training Scheduler.

This module contains the training scheduler for the unified AI loop:
- TrainingScheduler: Schedules and manages training runs with cluster-wide coordination

Extracted from unified_ai_loop.py for better modularity (Phase 2 refactoring).
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .config import (
    INITIAL_ELO_RATING,
    DataEvent,
    DataEventType,
    FeedbackConfig,
    FeedbackState,
    TrainingConfig,
)

if TYPE_CHECKING:
    from unified_ai_loop import ConfigPriorityQueue, EventBus, UnifiedLoopState

    from app.execution.backends import OrchestratorBackend
    from app.integration.pipeline_feedback import PipelineFeedbackController

import contextlib

from app.utils.paths import AI_SERVICE_ROOT

# Feature flag: disable local tasks when running on dedicated hosts
DISABLE_LOCAL_TASKS = os.environ.get("RINGRIFT_DISABLE_LOCAL_TASKS", "").lower() in ("1", "true", "yes")

# Optional ELO service import
try:
    from app.training.elo_service import get_elo_service
    HAS_ELO_SERVICE = True
except ImportError:
    HAS_ELO_SERVICE = False
    get_elo_service = None

# Optional Prometheus metrics - avoid duplicate registration
try:
    from prometheus_client import REGISTRY, Counter, Gauge, Histogram
    HAS_PROMETHEUS = True

    def _get_or_create_counter(name, desc, labels=None):
        if name in REGISTRY._names_to_collectors:
            return REGISTRY._names_to_collectors[name]
        return Counter(name, desc, labels or [])

    def _get_or_create_gauge(name, desc, labels=None):
        if name in REGISTRY._names_to_collectors:
            return REGISTRY._names_to_collectors[name]
        return Gauge(name, desc, labels or [])

    def _get_or_create_histogram(name, desc, labels=None, buckets=None):
        if name in REGISTRY._names_to_collectors:
            return REGISTRY._names_to_collectors[name]
        return Histogram(name, desc, labels or [], buckets=buckets)

    # Data quality metrics
    DATA_QUALITY_BLOCKED_TRAINING = _get_or_create_counter(
        'ringrift_data_quality_blocked_training_total',
        'Training runs blocked by data quality gate',
        ['reason']
    )

    # Training duration metrics
    TRAINING_DURATION_SECONDS = _get_or_create_histogram(
        'ringrift_training_duration_seconds',
        'Duration of training runs in seconds',
        ['config', 'result'],
        buckets=[60, 300, 600, 1800, 3600, 7200, 14400, 28800]
    )

    # Training retry metrics
    TRAINING_RETRY_ATTEMPTS = _get_or_create_counter(
        'ringrift_training_retry_attempts_total',
        'Number of training retry attempts',
        ['config']
    )
    TRAINING_RETRY_SUCCESS = _get_or_create_counter(
        'ringrift_training_retry_success_total',
        'Successful training runs after retry',
        ['config', 'attempt_number']
    )

    # Validation metrics
    VALIDATION_ERRORS = _get_or_create_counter(
        'ringrift_validation_errors_total',
        'Number of validation errors by type',
        ['error_type', 'config']
    )
    PARITY_FAILURE_RATE = _get_or_create_gauge(
        'ringrift_parity_failure_rate',
        'Current parity failure rate (0-1)',
        ['config']
    )

    # Lifecycle metrics
    LIFECYCLE_MAINTENANCE_RUNS = _get_or_create_counter(
        'ringrift_lifecycle_maintenance_runs_total',
        'Number of lifecycle maintenance runs',
        ['config']
    )
    MODELS_ARCHIVED = _get_or_create_counter(
        'ringrift_models_archived_total',
        'Number of models archived by lifecycle manager',
        ['config']
    )
    MODELS_DELETED = _get_or_create_counter(
        'ringrift_models_deleted_total',
        'Number of models deleted by lifecycle manager',
        ['config']
    )

except ImportError:
    HAS_PROMETHEUS = False
    DATA_QUALITY_BLOCKED_TRAINING = None
    TRAINING_DURATION_SECONDS = None
    TRAINING_RETRY_ATTEMPTS = None
    TRAINING_RETRY_SUCCESS = None
    VALIDATION_ERRORS = None
    PARITY_FAILURE_RATE = None
    LIFECYCLE_MAINTENANCE_RUNS = None
    MODELS_ARCHIVED = None
    MODELS_DELETED = None

# Improvement optimizer for positive feedback acceleration
try:
    from app.training.improvement_optimizer import (
        get_improvement_optimizer,
        should_fast_track_training,
    )
    HAS_IMPROVEMENT_OPTIMIZER = True
except ImportError:
    HAS_IMPROVEMENT_OPTIMIZER = False
    get_improvement_optimizer = None
    should_fast_track_training = None

# Resource optimizer for utilization-based scheduling
try:
    from app.coordination.resource_optimizer import get_utilization_status
    HAS_RESOURCE_OPTIMIZER = True
except ImportError:
    HAS_RESOURCE_OPTIMIZER = False
    get_utilization_status = None

# Coordination utilities
try:
    from app.coordination import estimate_task_duration
    HAS_COORDINATION = True
except ImportError:
    HAS_COORDINATION = False
    estimate_task_duration = None

# Pre-spawn health checking
try:
    from app.distributed.pre_spawn_health import gate_on_cluster_health
    HAS_PRE_SPAWN_HEALTH = True
except ImportError:
    HAS_PRE_SPAWN_HEALTH = False
    gate_on_cluster_health = None

# Feedback accelerator for momentum-based training
try:
    from app.integration.feedback_accelerator import (
        get_feedback_accelerator,
        record_games_generated,
    )
    HAS_FEEDBACK_ACCELERATOR = True
except ImportError:
    HAS_FEEDBACK_ACCELERATOR = False
    get_feedback_accelerator = None
    record_games_generated = None

# Value calibration
try:
    from app.training.value_calibration import CalibrationTracker
    HAS_VALUE_CALIBRATION = True
except ImportError:
    HAS_VALUE_CALIBRATION = False
    CalibrationTracker = None

# Temperature scheduling
try:
    from app.ai.temperature_schedule import create_temp_scheduler
    HAS_TEMPERATURE_SCHEDULING = True
except ImportError:
    HAS_TEMPERATURE_SCHEDULING = False
    create_temp_scheduler = None

# Simplified training triggers (2024-12)
try:
    from app.training.training_triggers import (
        TrainingTriggers,
        TriggerConfig,
        TriggerDecision,
    )
    HAS_SIMPLIFIED_TRIGGERS = True
except ImportError:
    HAS_SIMPLIFIED_TRIGGERS = False
    TrainingTriggers = None
    TriggerConfig = None
    TriggerDecision = None

# Curriculum feedback loop (2024-12)
try:
    from app.training.curriculum_feedback import (
        CurriculumFeedback,
        get_curriculum_feedback,
        get_curriculum_weights,
        record_selfplay_game,
    )
    HAS_CURRICULUM_FEEDBACK = True
except ImportError:
    HAS_CURRICULUM_FEEDBACK = False
    CurriculumFeedback = None
    get_curriculum_feedback = None
    record_selfplay_game = None
    get_curriculum_weights = None

# Advanced training utilities (2025-12)
try:
    from app.training.advanced_training import (
        AdaptivePrecisionManager,
        CMAESAutoTuner,
        GradientCheckpointing,
        LRFinder,
        LRFinderResult,
        OpponentStats,
        PFSPOpponentPool,
        PlateauConfig,
        ProgressiveLayerUnfreezing,
        SmartCheckpointManager,
        StabilityMetrics,
        SWAWithRestarts,
        # Phase 4 imports
        TrainingStabilityMonitor,
        create_phase4_training_suite,
    )
    HAS_ADVANCED_TRAINING = True
except ImportError:
    HAS_ADVANCED_TRAINING = False
    LRFinder = None
    LRFinderResult = None
    GradientCheckpointing = None
    PFSPOpponentPool = None
    OpponentStats = None
    CMAESAutoTuner = None
    PlateauConfig = None
    # Phase 4 fallbacks
    TrainingStabilityMonitor = None
    StabilityMetrics = None
    AdaptivePrecisionManager = None
    ProgressiveLayerUnfreezing = None
    SWAWithRestarts = None
    SmartCheckpointManager = None
    create_phase4_training_suite = None

# Streaming pipeline integration (2025-12 bottleneck fixes)
try:
    from app.training.streaming_pipeline import (
        GameSample,
        StreamingConfig,
        StreamingDataPipeline,
    )
    HAS_STREAMING_PIPELINE = True
except ImportError:
    HAS_STREAMING_PIPELINE = False
    StreamingDataPipeline = None
    StreamingConfig = None
    GameSample = None

# Async shadow validation (2025-12 bottleneck fixes)
try:
    from app.ai.shadow_validation import (
        AsyncShadowValidator,
        create_async_shadow_validator,
    )
    HAS_ASYNC_VALIDATION = True
except ImportError:
    HAS_ASYNC_VALIDATION = False
    AsyncShadowValidator = None
    create_async_shadow_validator = None

# Connection pool for unified WAL (2025-12 bottleneck fixes)
try:
    from app.distributed.unified_wal import ConnectionPool
    HAS_CONNECTION_POOL = True
except ImportError:
    HAS_CONNECTION_POOL = False

# Enhanced fault tolerance with retry policies and circuit breakers (2025-12)
try:
    from app.training.fault_tolerance import RetryPolicy
    HAS_RETRY_POLICY = True
except ImportError:
    HAS_RETRY_POLICY = False
    RetryPolicy = None

# Circuit breaker for training operations (2025-12)
try:
    from app.distributed.circuit_breaker import (
        CircuitBreakerRegistry,
        CircuitState,
    )
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False
    CircuitBreakerRegistry = None
    CircuitState = None

# NNUE dataset validation (2025-12)
try:
    from app.training.nnue_dataset import (
        DataValidationResult,
        validate_database_integrity,
        validate_nnue_dataset,
    )
    HAS_NNUE_VALIDATION = True
except ImportError:
    HAS_NNUE_VALIDATION = False
    validate_nnue_dataset = None
    validate_database_integrity = None
    DataValidationResult = None
    ConnectionPool = None

# Elo-based checkpoint selection (2025-12)
try:
    from scripts.select_best_checkpoint_by_elo import select_best_checkpoint
    HAS_ELO_CHECKPOINT_SELECTION = True
except ImportError:
    HAS_ELO_CHECKPOINT_SELECTION = False
    select_best_checkpoint = None

# Data manifest for pre-training sync (quality-aware data sync)
try:
    from app.distributed.unified_manifest import DataManifest
    HAS_DATA_MANIFEST = True
except ImportError:
    HAS_DATA_MANIFEST = False
    DataManifest = None

# Automated post-training promotion (2026-01)
try:
    from app.training.auto_promotion import evaluate_and_promote
    HAS_AUTO_PROMOTION = True
except ImportError:
    HAS_AUTO_PROMOTION = False
    evaluate_and_promote = None


class TrainingScheduler:
    """Schedules and manages training runs with cluster-wide coordination."""

    def __init__(
        self,
        config: TrainingConfig,
        state: UnifiedLoopState,
        event_bus: EventBus,
        feedback_config: FeedbackConfig | None = None,
        feedback: PipelineFeedbackController | None = None,
        config_priority: ConfigPriorityQueue | None = None
    ):
        # Import ConfigPriorityQueue at runtime to avoid circular imports
        from scripts.unified_ai_loop import ConfigPriorityQueue as CPQ

        self.config = config
        self.state = state
        self.event_bus = event_bus
        self.feedback_config = feedback_config or FeedbackConfig()
        self.feedback = feedback
        self.config_priority = config_priority or CPQ()
        self._training_process: asyncio.subprocess.Process | None = None
        # Dynamic threshold tracking (for promotion velocity calculation)
        self._promotion_history: list[float] = []  # Timestamps of recent promotions
        self._training_history: list[float] = []   # Timestamps of recent training runs
        # Per-config training coordination (allows parallel training of different configs)
        self._training_locks: dict[str, int] = {}  # config_key -> fd
        self._training_lock_paths: dict[str, Path] = {}
        self._max_concurrent_training = 8  # Feb 2026: Lowered from 20 to leave capacity for selfplay
        # Calibration tracking (per config)
        self._calibration_trackers: dict[str, Any] = {}
        if HAS_VALUE_CALIBRATION:
            for config_key in state.configs:
                self._calibration_trackers[config_key] = CalibrationTracker(window_size=1000)
        # Temperature scheduler for self-play exploration
        self._temp_scheduler: Any | None = None
        if HAS_TEMPERATURE_SCHEDULING:
            self._temp_scheduler = create_temp_scheduler(state.current_temperature_preset or "default")

        # Simplified 3-signal trigger system (2024-12)
        self._simplified_triggers: Any | None = None
        if HAS_SIMPLIFIED_TRIGGERS and getattr(config, 'use_simplified_triggers', True):
            trigger_cfg = TriggerConfig(
                freshness_threshold=config.trigger_threshold_games,
                staleness_hours=getattr(config, 'staleness_hours', 6.0),
                min_win_rate=getattr(config, 'min_win_rate_threshold', 0.45),
                min_interval_minutes=config.min_interval_seconds / 60,
                max_concurrent_training=self._max_concurrent_training,
                bootstrap_threshold=getattr(config, 'bootstrap_threshold', 50),
            )
            self._simplified_triggers = TrainingTriggers(trigger_cfg)
            print("[Training] Using simplified 3-signal trigger system")

        # Curriculum feedback loop (2024-12)
        self._curriculum_feedback: Any | None = None
        if HAS_CURRICULUM_FEEDBACK:
            self._curriculum_feedback = get_curriculum_feedback()
            print("[Training] Using curriculum feedback loop for adaptive weights")

        # Advanced training utilities (2025-12)
        self._pfsp_pool: Any | None = None
        self._cmaes_auto_tuner: Any | None = None
        self._gradient_checkpointing: Any | None = None

        if HAS_ADVANCED_TRAINING:
            # Initialize PFSP opponent pool
            if getattr(config, 'use_pfsp', True):
                self._pfsp_pool = PFSPOpponentPool(
                    max_pool_size=getattr(config, 'pfsp_max_pool_size', 20),
                    hard_opponent_weight=getattr(config, 'pfsp_hard_opponent_weight', 0.7),
                    diversity_weight=getattr(config, 'pfsp_diversity_weight', 0.2),
                    recency_weight=getattr(config, 'pfsp_recency_weight', 0.1),
                )
                print("[Training] PFSP opponent pool initialized")

            # Initialize CMA-ES auto-tuner (per config)
            self._cmaes_auto_tuners: dict[str, Any] = {}
            if getattr(config, 'use_cmaes_auto_tuning', True):
                for config_key in state.configs:
                    parts = config_key.rsplit("_", 1)
                    board_type = parts[0]
                    num_players = int(parts[1].replace("p", "")) if len(parts) > 1 else 2
                    plateau_cfg = PlateauConfig(
                        patience=getattr(config, 'cmaes_plateau_patience', 10),
                    )
                    self._cmaes_auto_tuners[config_key] = CMAESAutoTuner(
                        board_type=board_type,
                        num_players=num_players,
                        plateau_config=plateau_cfg,
                        min_epochs_between_tuning=getattr(config, 'cmaes_min_epochs_between', 50),
                        max_auto_tunes=getattr(config, 'cmaes_max_auto_tunes', 3),
                    )
                print(f"[Training] CMA-ES auto-tuners initialized for {len(self._cmaes_auto_tuners)} configs")

        # Elo-based checkpoint selection configuration (2025-12)
        self._use_elo_checkpoint_selection = (
            HAS_ELO_CHECKPOINT_SELECTION and
            getattr(config, 'use_elo_checkpoint_selection', True)
        )
        self._elo_selection_games_per_opponent = getattr(config, 'elo_selection_games', 10)
        self._elo_selection_copy_best = getattr(config, 'elo_selection_copy_best', True)
        if self._use_elo_checkpoint_selection:
            print("[Training] Elo-based checkpoint selection enabled")

        # Bottleneck fix integrations (2025-12)
        self._streaming_pipelines: dict[str, Any] = {}
        self._async_validator: Any | None = None
        self._connection_pool: Any | None = None
        self._parity_failure_rate: float = 0.0  # Track parity failures for training decisions
        # Execution backend for remote training dispatch (coordinator mode)
        self._backend: OrchestratorBackend | None = None
        # Auto-recovery state (Phase 7)
        self._retry_attempts: dict[str, int] = {}  # config_key -> retry count
        self._last_failure_time: dict[str, float] = {}  # config_key -> timestamp
        self._pending_retries: list[tuple[str, float]] = []  # (config_key, scheduled_time)

        # Initialize streaming pipelines for each config
        if HAS_STREAMING_PIPELINE and getattr(config, 'use_streaming_pipeline', True):
            for config_key in state.configs:
                parts = config_key.rsplit("_", 1)
                board_type = parts[0]
                num_players = int(parts[1].replace("p", "")) if len(parts) > 1 else 2
                db_path = config.selfplay_db_path / f"{config_key}.db"
                if db_path.exists():
                    streaming_cfg = StreamingConfig(
                        poll_interval_seconds=getattr(config, 'streaming_poll_interval', 5.0),
                        buffer_size=getattr(config, 'streaming_buffer_size', 10000),
                        dedupe_enabled=True,
                        priority_sampling=True,
                    )
                    self._streaming_pipelines[config_key] = StreamingDataPipeline(
                        db_path=db_path,
                        board_type=board_type,
                        num_players=num_players,
                        config=streaming_cfg,
                    )
            if self._streaming_pipelines:
                print(f"[Training] Streaming pipelines initialized for {len(self._streaming_pipelines)} configs")

        # Initialize async shadow validator
        if HAS_ASYNC_VALIDATION and getattr(config, 'use_async_validation', True):
            self._async_validator = create_async_shadow_validator(
                sample_rate=getattr(config, 'validation_sample_rate', 0.05),
                enabled=True,
            )
            if self._async_validator:
                print("[Training] Async shadow validation enabled (non-blocking)")

        # Initialize connection pool for database operations
        if HAS_CONNECTION_POOL and getattr(config, 'use_connection_pool', True):
            wal_db_path = AI_SERVICE_ROOT / "data" / "unified_wal.db"
            if wal_db_path.parent.exists():
                self._connection_pool = ConnectionPool(wal_db_path)
                print("[Training] Connection pool enabled for database operations")

        # Circuit breaker for training operations (2025-12)
        self._circuit_breaker: Any | None = None
        if HAS_CIRCUIT_BREAKER:
            # Use the singleton registry and get/create a breaker for training operations
            self._circuit_breaker = CircuitBreakerRegistry.get_instance().get_breaker("training")
            print("[Training] Circuit breaker protection enabled for training spawn")

        # Retry policy configuration (2025-12)
        # RetryPolicy provides dicts with: max_retries, base_delay, max_delay, exponential_base
        self._retry_policy: dict[str, Any] | None = None
        if HAS_RETRY_POLICY:
            # Use CONSERVATIVE policy for training (important operations)
            self._retry_policy = RetryPolicy.CONSERVATIVE
            print(f"[Training] Using CONSERVATIVE retry policy (max_retries={self._retry_policy['max_retries']})")

        # Data sync service reference for pre-training sync (P5)
        self._sync_service: Any | None = None
        self._manifest: Any | None = None
        self._pre_training_sync_enabled = getattr(config, 'pre_training_sync', True)
        self._pre_training_sync_min_quality = getattr(config, 'pre_training_sync_min_quality', 0.6)
        self._pre_training_sync_limit = getattr(config, 'pre_training_sync_limit', 500)
        # Track games consumed per training run for quality feedback
        self._training_consumed_games: dict[str, list[str]] = {}  # config_key -> [game_ids]

        # Initialize manifest directly for pre-training sync (quality-aware data sync)
        if HAS_DATA_MANIFEST and self._pre_training_sync_enabled:
            manifest_path = AI_SERVICE_ROOT / "data" / "data_manifest.db"
            if manifest_path.parent.exists():
                try:
                    self._manifest = DataManifest(manifest_path)
                    print(f"[Training] Data manifest initialized for pre-training sync: {manifest_path}")
                except Exception as e:
                    print(f"[Training] Warning: Failed to initialize data manifest: {e}")

    def set_sync_service(self, sync_service: Any) -> None:
        """Set reference to data sync service for pre-training sync.

        Args:
            sync_service: UnifiedDataSyncService instance or object with manifest attribute
        """
        self._sync_service = sync_service
        if sync_service and hasattr(sync_service, 'manifest'):
            self._manifest = sync_service.manifest
            print("[Training] Data sync service connected for pre-training sync")

    async def _sync_high_quality_data_before_training(self, config_key: str) -> int:
        """Sync high-quality games from cluster before training.

        Uses the manifest's priority queue to identify and sync
        games with quality_score >= min_quality threshold.

        Args:
            config_key: Training configuration key

        Returns:
            Number of games synced
        """
        if not self._manifest or not self._pre_training_sync_enabled:
            return 0

        try:
            # Get high-quality games from priority queue
            pending = self._manifest.get_priority_queue_batch(
                limit=self._pre_training_sync_limit,
                min_priority=self._pre_training_sync_min_quality,
            )

            if not pending:
                return 0

            print(f"[Training] Pre-training sync: {len(pending)} high-quality games pending")

            # Mark games as consumed for training and track IDs
            synced = 0
            entry_ids = []
            game_ids = []
            for entry in pending:
                entry_ids.append(entry.id)
                game_ids.append(entry.game_id)
                synced += 1

            if entry_ids:
                self._manifest.mark_queue_entries_synced(entry_ids)
                # Track consumed games for quality feedback
                self._training_consumed_games[config_key] = game_ids
                print(f"[Training] Pre-training sync complete: {synced} high-quality games "
                      f"(avg priority: {sum(e.priority_score for e in pending)/len(pending):.2f})")

            return synced

        except Exception as e:
            print(f"[Training] Pre-training sync error: {e}")
            return 0

    async def _update_source_quality_after_training(
        self, config_key: str, training_result: dict[str, Any]
    ) -> None:
        """Update source quality in manifest based on training results.

        If training improved Elo, boost the quality score of games that
        contributed to this training. If Elo regressed, reduce quality.

        Args:
            config_key: Training configuration key
            training_result: Dict with training results including elo_feedback
        """
        if not self._manifest:
            return

        try:
            # Extract Elo delta from training result
            elo_feedback = training_result.get("elo_feedback", {})
            elo_delta = elo_feedback.get("avg_elo_delta", 0.0)
            elo_regression = elo_feedback.get("elo_regression", False)

            # Compute quality adjustment based on Elo change
            # Positive Elo = good quality data, negative = poor quality
            if elo_delta > 10:
                quality_boost = min(0.1, elo_delta / 100)  # Max 10% boost
            elif elo_regression or elo_delta < -10:
                quality_boost = max(-0.1, elo_delta / 100)  # Max 10% penalty
            else:
                quality_boost = 0.0  # Neutral

            if abs(quality_boost) < 0.01:
                return  # No significant change

            # Get recently synced games (these contributed to training)
            # Note: We already marked them synced in pre-training, now we update quality
            stats = self._manifest.get_priority_queue_stats()
            if stats.get("total", 0) > 0:
                print(f"[Training] Quality feedback: Elo delta={elo_delta:.1f}, "
                      f"adjusting source quality by {quality_boost:+.2f}")

                # Update quality distribution tracking in manifest
                # Future: could track per-source quality adjustments
                self._manifest.cleanup_old_queue_entries(days=7)

        except Exception as e:
            print(f"[Training] Error in source quality update: {e}")

    def get_training_quality_stats(self, config_key: str) -> dict[str, Any]:
        """Get quality statistics for a training configuration.

        Returns stats about games consumed, quality distribution, and manifest state.

        Args:
            config_key: Training configuration key

        Returns:
            Dict with quality statistics
        """
        stats = {
            "config_key": config_key,
            "consumed_games_count": len(self._training_consumed_games.get(config_key, [])),
            "pre_training_sync_enabled": self._pre_training_sync_enabled,
            "min_quality_threshold": self._pre_training_sync_min_quality,
        }

        if self._manifest:
            try:
                queue_stats = self._manifest.get_priority_queue_stats()
                stats["priority_queue"] = queue_stats
                quality_dist = self._manifest.get_quality_distribution()
                stats["quality_distribution"] = quality_dist
            except Exception as e:
                stats["manifest_error"] = str(e)

        return stats

    def _get_dynamic_threshold(self, config_key: str) -> int:
        """Calculate dynamic training threshold based on promotion velocity."""
        base_threshold = self.config.trigger_threshold_games  # Default: 500

        now = time.time()
        recent_promotions = [t for t in self._promotion_history if now - t < 3600 * 6]
        promotions_per_hour = len(recent_promotions) / 6.0 if recent_promotions else 0

        recent_training = [t for t in self._training_history if now - t < 3600 * 6]
        training_per_hour = len(recent_training) / 6.0 if recent_training else 0

        adjustment = 1.0

        if promotions_per_hour > 0.5:
            adjustment *= 0.7
        elif promotions_per_hour < 0.1:
            adjustment *= 0.8
        else:
            adjustment *= 0.9

        if training_per_hour > 2 and promotions_per_hour < 0.2:
            adjustment *= 1.5
            print(f"[Training] Dynamic: High training ({training_per_hour:.1f}/hr) but low promotion ({promotions_per_hour:.1f}/hr) - increasing threshold")

        config_state = self.state.configs.get(config_key)
        if config_state:
            time_since_promotion = now - config_state.last_promotion_time
            if time_since_promotion > 3600 * 2:
                adjustment *= 0.75
                print(f"[Training] Dynamic: {time_since_promotion/3600:.1f}h since last promotion for {config_key} - lowering threshold")

        dynamic_threshold = int(base_threshold * adjustment)
        min_threshold = base_threshold // 4
        max_threshold = base_threshold * 2

        final_threshold = max(min_threshold, min(max_threshold, dynamic_threshold))

        # Improvement optimizer acceleration
        if HAS_IMPROVEMENT_OPTIMIZER:
            try:
                optimizer = get_improvement_optimizer()
                optimizer_threshold = optimizer.get_dynamic_threshold(config_key)
                metrics = optimizer.get_improvement_metrics()

                if optimizer_threshold < final_threshold:
                    streak_info = f"streak={metrics.get('consecutive_promotions', 0)}"
                    print(f"[ImprovementOptimizer] Accelerating threshold for {config_key}: "
                          f"{final_threshold} → {optimizer_threshold} ({streak_info})")
                    final_threshold = optimizer_threshold

                if should_fast_track_training(config_key):
                    fast_threshold = max(min_threshold, final_threshold * 8 // 10)
                    if fast_threshold < final_threshold:
                        print(f"[ImprovementOptimizer] Fast-tracking {config_key}: {final_threshold} → {fast_threshold}")
                        final_threshold = fast_threshold
            except Exception as e:
                if self.config.verbose:
                    print(f"[ImprovementOptimizer] Error getting threshold: {e}")

        # Utilization-based adjustment
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
            except (AttributeError, KeyError, TypeError, ValueError):
                pass

        # Underrepresented config priority
        if hasattr(self.config_priority, '_trained_model_counts'):
            self.config_priority._update_trained_model_counts()
            model_count = self.config_priority._trained_model_counts.get(config_key, 0)

            if model_count == 0:
                bootstrap_threshold = 50
                if final_threshold > bootstrap_threshold:
                    print(f"[Training] BOOTSTRAP: {config_key} has 0 trained models - threshold {final_threshold} → {bootstrap_threshold}")
                    final_threshold = bootstrap_threshold
            elif model_count == 1:
                single_model_threshold = min_threshold
                if final_threshold > single_model_threshold:
                    print(f"[Training] UNDERREPRESENTED: {config_key} has 1 model - threshold {final_threshold} → {single_model_threshold}")
                    final_threshold = single_model_threshold
            elif model_count <= 3:
                catchup_threshold = min_threshold * 3 // 2
                if final_threshold > catchup_threshold:
                    print(f"[Training] CATCHUP: {config_key} has {model_count} models - threshold {final_threshold} → {catchup_threshold}")
                    final_threshold = catchup_threshold

        # Phase 2.4: Win rate → training feedback
        # Adjust threshold based on evaluation win rates to avoid redundant training
        if config_state:
            win_rate = getattr(config_state, 'win_rate', 0.5)
            win_rate_trend = getattr(config_state, 'win_rate_trend', 0.0)
            consecutive_high = getattr(config_state, 'consecutive_high_win_rate', 0)

            # High win rate (>70%) consistently → de-prioritize training (already strong)
            if consecutive_high >= 3 and win_rate > 0.7:
                skip_factor = 1.5 + (consecutive_high - 3) * 0.2  # Max ~2.5x
                skip_factor = min(skip_factor, 2.5)
                old_threshold = final_threshold
                final_threshold = min(max_threshold, int(final_threshold * skip_factor))
                if final_threshold > old_threshold:
                    print(f"[Training] WIN_RATE_SKIP: {config_key} has {win_rate:.1%} win rate "
                          f"(consecutive_high={consecutive_high}) - threshold {old_threshold} → {final_threshold}")

            # Low win rate (<50%) or declining → prioritize training urgently
            elif win_rate < 0.5 or win_rate_trend < -0.05:
                urgency_factor = 0.6 if win_rate < 0.4 else 0.8
                old_threshold = final_threshold
                final_threshold = max(min_threshold, int(final_threshold * urgency_factor))
                if final_threshold < old_threshold:
                    trend_str = f", trend={win_rate_trend:+.1%}" if win_rate_trend < -0.05 else ""
                    print(f"[Training] WIN_RATE_URGENT: {config_key} has {win_rate:.1%} win rate{trend_str} "
                          f"- threshold {old_threshold} → {final_threshold}")

            # Declining win rate but still decent → moderate priority increase
            elif win_rate_trend < -0.02 and win_rate < 0.65:
                old_threshold = final_threshold
                final_threshold = max(min_threshold, int(final_threshold * 0.9))
                if final_threshold < old_threshold:
                    print(f"[Training] WIN_RATE_DECLINING: {config_key} win rate declining ({win_rate_trend:+.1%}) "
                          f"- threshold {old_threshold} → {final_threshold}")

        # Curriculum feedback weight adjustment (2024-12)
        # Applies weight from recent selfplay performance (0.5 to 2.0)
        curriculum_weight = self.get_curriculum_weight(config_key)
        if curriculum_weight != 1.0:
            old_threshold = final_threshold
            # Weight > 1.0 = needs more training (lower threshold)
            # Weight < 1.0 = already strong (higher threshold)
            final_threshold = int(final_threshold / curriculum_weight)
            final_threshold = max(min_threshold, min(max_threshold, final_threshold))
            if final_threshold != old_threshold:
                print(f"[Training] CURRICULUM_WEIGHT: {config_key} weight={curriculum_weight:.2f} "
                      f"- threshold {old_threshold} → {final_threshold}")

        # Parity failure rate adjustment (2025-12)
        # High parity failures indicate data quality issues - be more conservative
        if self._parity_failure_rate > 0.05:
            old_threshold = final_threshold
            # Scale up threshold based on failure rate: up to ~1.2x at 10% failure rate
            parity_factor = 1.0 + (self._parity_failure_rate * 2.0)
            final_threshold = min(max_threshold, int(final_threshold * parity_factor))
            if final_threshold > old_threshold:
                print(f"[Training] PARITY_CAUTION: {config_key} parity failure rate={self._parity_failure_rate:.1%} "
                      f"- threshold {old_threshold} → {final_threshold}")

        if final_threshold != base_threshold:
            print(f"[Training] Dynamic threshold for {config_key}: {final_threshold} (base: {base_threshold}, adj: {adjustment:.2f})")

        return final_threshold

    def should_train_simplified(
        self,
        config_key: str,
        games_since_training: int,
        win_rate: float = 0.5,
        model_count: int = 1,
    ) -> tuple[bool, str, float]:
        """Check if training should run using simplified 3-signal system.

        Args:
            config_key: Config identifier (e.g., "square8_2p")
            games_since_training: Number of new games since last training
            win_rate: Current win rate (0.0 to 1.0)
            model_count: Number of trained models for this config

        Returns:
            Tuple of (should_train, reason, priority)
        """
        if self._simplified_triggers is None:
            # Fall back to legacy system
            threshold = self._get_dynamic_threshold(config_key)
            should = games_since_training >= threshold
            return should, f"games={games_since_training} >= threshold={threshold}", float(games_since_training)

        # Update state in simplified triggers
        self._simplified_triggers.update_config_state(
            config_key,
            games_count=games_since_training + self._simplified_triggers.get_config_state(config_key).last_training_games,
            win_rate=win_rate,
            model_count=model_count,
        )

        # Get decision from simplified system
        decision = self._simplified_triggers.should_train(config_key)
        return decision.should_train, decision.reason, decision.priority

    def record_training_complete_simplified(self, config_key: str, games_at_training: int) -> None:
        """Record training completion in simplified trigger system."""
        if self._simplified_triggers is not None:
            self._simplified_triggers.record_training_complete(config_key, games_at_training)

    def get_next_training_config_simplified(self) -> str | None:
        """Get the highest priority config that should train using simplified system."""
        if self._simplified_triggers is None:
            return None

        decision = self._simplified_triggers.get_next_training_config()
        if decision is not None:
            print(f"[Training] Simplified trigger: {decision.config_key} (priority={decision.priority:.2f}, reason={decision.reason})")
            return decision.config_key
        return None

    def record_promotion(self):
        """Record a successful promotion for velocity tracking."""
        now = time.time()
        self._promotion_history.append(now)
        self._promotion_history = [t for t in self._promotion_history if now - t < 86400]

    def record_training_start(self):
        """Record a training run start for velocity tracking."""
        now = time.time()
        self._training_history.append(now)
        self._training_history = [t for t in self._training_history if now - t < 86400]

    def record_selfplay_result(
        self,
        config_key: str,
        winner: int,
        model_elo: float = INITIAL_ELO_RATING,
    ) -> None:
        """Record a selfplay game result for curriculum feedback.

        Args:
            config_key: Config identifier (e.g., "square8_2p")
            winner: 1 = model won, -1 = model lost, 0 = draw
            model_elo: Current model Elo rating
        """
        if self._curriculum_feedback is not None:
            self._curriculum_feedback.record_game(
                config_key, winner, model_elo, opponent_type="selfplay"
            )

    def get_curriculum_weights(self) -> dict[str, float]:
        """Get curriculum weights for all configs.

        Returns:
            Dict mapping config_key → weight (0.5 to 2.0)
            Higher weight = more training attention needed
        """
        if self._curriculum_feedback is not None:
            return self._curriculum_feedback.get_curriculum_weights()
        return {}

    def get_curriculum_weight(self, config_key: str) -> float:
        """Get curriculum weight for a specific config.

        Returns:
            Weight between 0.5 (de-prioritize) and 2.0 (high priority)
        """
        weights = self.get_curriculum_weights()
        return weights.get(config_key, 1.0)

    def get_training_quality(self, config_key: str) -> dict[str, Any]:
        """Get training quality metrics for feedback to selfplay.

        This enables the selfplay generator to adjust engine selection
        based on training health - using more exploratory engines when
        training shows signs of overfitting or plateau.

        Args:
            config_key: Config identifier (e.g., "square8_2p")

        Returns:
            Dict with quality metrics:
            - loss_plateau: True if training loss not improving
            - overfit_detected: True if train/val divergence detected
            - last_val_loss: Most recent validation loss
            - parity_failure_rate: Rate of parity validation failures
        """
        quality: dict[str, Any] = {
            'loss_plateau': False,
            'overfit_detected': False,
            'last_val_loss': None,
            'parity_failure_rate': self._parity_failure_rate,
        }

        # Check CMA-ES auto-tuner for plateau detection
        if hasattr(self, '_cmaes_auto_tuners') and config_key in self._cmaes_auto_tuners:
            tuner = self._cmaes_auto_tuners[config_key]
            if hasattr(tuner, 'is_plateau_detected'):
                quality['loss_plateau'] = tuner.is_plateau_detected()

        # Check simplified triggers for regression signals
        if self._simplified_triggers is not None:
            state = self._simplified_triggers.get_config_state(config_key)
            if hasattr(state, 'regression_detected'):
                # Regression can indicate overfitting
                quality['overfit_detected'] = state.regression_detected

        # Check streaming pipeline for data quality issues
        if hasattr(self, '_streaming_pipelines') and config_key in self._streaming_pipelines:
            pipeline = self._streaming_pipelines[config_key]
            if hasattr(pipeline, 'get_stats'):
                stats = pipeline.get_stats()
                if stats and stats.get('duplicate_rate', 0) > 0.5:
                    # High duplicate rate suggests need for more diverse data
                    quality['overfit_detected'] = True

        return quality

    def record_training_complete_for_curriculum(self, config_key: str) -> None:
        """Record training completion for curriculum feedback."""
        if self._curriculum_feedback is not None:
            self._curriculum_feedback.record_training(config_key)

    def set_feedback_controller(self, feedback: PipelineFeedbackController):
        """Set the feedback controller (called after initialization)."""
        self.feedback = feedback

    def set_execution_backend(self, backend: OrchestratorBackend):
        """Set the execution backend for remote training dispatch."""
        self._backend = backend

    async def _dispatch_remote_training(self, config_key: str) -> bool:
        """Dispatch training to a remote GPU worker via execution backend.

        Args:
            config_key: Configuration key (e.g., "square8_2p")

        Returns:
            True if training was successfully dispatched
        """
        if self._backend is None:
            print("[Training] No execution backend for remote training")
            return False

        parts = config_key.rsplit("_", 1)
        board_type = parts[0]
        num_players = int(parts[1].replace("p", "")) if len(parts) > 1 else 2

        # Find training data path - check JSONL files first, then DB files
        games_dir = AI_SERVICE_ROOT / "data" / "games"
        data_path = None

        # JSONL sources (in order of preference)
        jsonl_sources = [
            games_dir / "gpu_selfplay" / config_key / "games.jsonl",
            games_dir / f"{config_key}.jsonl",
            AI_SERVICE_ROOT / "data" / "selfplay" / f"gpu_{config_key}" / "games.jsonl",
        ]

        # Also search cluster sync directories for matching JSONL
        for subdir in ["cluster", "cluster_sync"]:
            cluster_dir = games_dir / subdir
            if cluster_dir.exists():
                # Find JSONL files that match the config key
                for jsonl_file in cluster_dir.rglob("*.jsonl"):
                    if config_key in jsonl_file.name or board_type in jsonl_file.name:
                        jsonl_sources.append(jsonl_file)

        # Check JSONL sources
        for src in jsonl_sources:
            if isinstance(src, Path) and src.exists() and src.stat().st_size > 0:
                data_path = str(src)
                print(f"[Training] Found JSONL data: {src.name}")
                break

        # If no JSONL, look for consolidated DB files
        if not data_path:
            db_sources = [
                games_dir / "cluster_synced.db",  # Main synced database
                games_dir / f"canonical_{board_type}.db",  # Canonical games
                games_dir / "consolidated_training_v2.db",  # Consolidated training data
                games_dir / f"{board_type}.db",  # Board-specific DB
            ]
            for src in db_sources:
                if src.exists() and src.stat().st_size > 0:
                    data_path = str(src)
                    print(f"[Training] Found DB data: {src.name}")
                    break

        if not data_path:
            print(f"[Training] No training data found for {config_key}")
            # Schedule retry in case data is still syncing
            self._schedule_retry(config_key, "no_data")
            return False

        # Output model path
        model_output = str(AI_SERVICE_ROOT / "models" / f"ringrift_{config_key}.pth")

        # Training epochs from config
        epochs = getattr(self.config, 'epochs', 100)

        print(f"[Training] Dispatching remote training for {config_key} via backend")

        try:
            result = await self._backend.run_training(
                data_path=data_path,
                model_output_path=model_output,
                epochs=epochs,
                board_type=board_type,
                num_players=num_players,
            )

            if result.success:
                print(f"[Training] Remote training completed on {result.worker}: {config_key}")
                # Emit training completed event
                await self.event_bus.publish(DataEvent(
                    event_type=DataEventType.TRAINING_COMPLETED,
                    payload={
                        "config": config_key,
                        "model_path": model_output,
                        "worker": result.worker,
                        "remote": True,
                    }
                ))

                # Run auto-promotion if enabled
                if HAS_AUTO_PROMOTION and getattr(self.config, 'auto_promote', False):
                    await self._run_auto_promotion(
                        model_path=model_output,
                        board_type=board_type,
                        num_players=num_players,
                        config_key=config_key,
                    )

                return True
            else:
                print(f"[Training] Remote training failed on {result.worker}: {result.error}")
                return False
        except Exception as e:
            print(f"[Training] Remote training dispatch error: {e}")
            return False

    async def _run_auto_promotion(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        config_key: str,
    ) -> bool:
        """Run automated promotion evaluation after training.

        Args:
            model_path: Path to the trained model
            board_type: Board type (e.g., "hex8")
            num_players: Number of players (2, 3, or 4)
            config_key: Configuration key (e.g., "hex8_2p")

        Returns:
            True if promotion succeeded, False otherwise
        """
        if not HAS_AUTO_PROMOTION or evaluate_and_promote is None:
            print(f"[AutoPromotion] Auto-promotion not available for {config_key}")
            return False

        games = getattr(self.config, 'auto_promote_games', 30)
        sync_to_cluster = getattr(self.config, 'auto_promote_sync', True)

        print(f"[AutoPromotion] Starting evaluation for {config_key} ({games} games per opponent)")

        try:
            result = await evaluate_and_promote(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
                games=games,
                sync_to_cluster=sync_to_cluster,
            )

            if result.approved:
                print(f"[AutoPromotion] Model PROMOTED for {config_key}: {result.reason}")
                # Emit promotion event
                await self.event_bus.publish(DataEvent(
                    event_type=DataEventType.MODEL_PROMOTED,
                    payload={
                        "config": config_key,
                        "model_path": model_path,
                        "promoted_path": result.promoted_path,
                        "reason": result.reason,
                        "criterion_met": result.decision.criterion_met.value if result.decision.criterion_met else None,
                        "estimated_elo": result.eval_results.estimated_elo if result.eval_results else None,
                        "auto_promoted": True,
                    }
                ))
                return True
            else:
                print(f"[AutoPromotion] Model NOT promoted for {config_key}: {result.reason}")
                return False

        except Exception as e:
            print(f"[AutoPromotion] Error evaluating {config_key}: {e}")
            return False

    def _acquire_training_lock(self, config_key: str | None = None) -> bool:
        """Acquire per-config training lock using file locking.

        Allows up to _max_concurrent_training parallel training runs.
        Different configs can train simultaneously (e.g., square8 + hexagonal).
        """
        import fcntl
        import socket

        # Check if we've hit the concurrent training limit
        active_locks = len(self._training_locks)
        if active_locks >= self._max_concurrent_training:
            print(f"[Training] Max concurrent training ({self._max_concurrent_training}) reached")
            return False

        lock_dir = AI_SERVICE_ROOT / "data" / "coordination"
        lock_dir.mkdir(parents=True, exist_ok=True)

        hostname = socket.gethostname()
        lock_name = f"training.{config_key or 'global'}.{hostname}.lock"
        lock_path = lock_dir / lock_name

        try:
            fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, f"{os.getpid()}\n".encode())

            self._training_locks[config_key or 'global'] = fd
            self._training_lock_paths[config_key or 'global'] = lock_path
            print(f"[Training] Acquired lock for {config_key or 'global'} on {hostname} ({active_locks + 1}/{self._max_concurrent_training} slots used)")
            return True
        except (OSError, BlockingIOError) as e:
            with contextlib.suppress(OSError):
                os.close(fd)
            print(f"[Training] Lock acquisition failed for {config_key}: {e}")
            return False

    def _release_training_lock(self, config_key: str | None = None):
        """Release the per-config training lock."""
        import fcntl

        key = config_key or 'global'
        fd = self._training_locks.get(key)
        lock_path = self._training_lock_paths.get(key)

        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                print(f"[Training] Released lock for {key}")
            except Exception as e:
                print(f"[Training] Error releasing lock for {key}: {e}")
            finally:
                self._training_locks.pop(key, None)
                self._training_lock_paths.pop(key, None)
            if lock_path and lock_path.exists():
                with contextlib.suppress(Exception):
                    lock_path.unlink()

    def is_training_locked_elsewhere(self, config_key: str | None = None) -> bool:
        """Check if training is running on another host for a specific config."""
        import socket

        lock_dir = AI_SERVICE_ROOT / "data" / "coordination"
        if not lock_dir.exists():
            return False

        hostname = socket.gethostname()
        # Look for locks matching this config (or any lock if config_key is None)
        pattern = f"training.{config_key}.*" if config_key else "training.*"

        for lock_file in lock_dir.glob(pattern + ".lock"):
            # Skip our own locks
            if hostname in lock_file.name:
                continue
            try:
                if lock_file.stat().st_size > 0:
                    age = time.time() - lock_file.stat().st_mtime
                    if age < 3600:
                        parts = lock_file.name.replace("training.", "").replace(".lock", "").split(".")
                        other_config = parts[0] if len(parts) > 1 else "unknown"
                        other_host = parts[-1] if len(parts) > 1 else parts[0]
                        print(f"[Training] Config {other_config} locked by {other_host}")
                        return True
            except (OSError, FileNotFoundError, PermissionError):
                continue
        return False

    def count_active_training_runs(self) -> int:
        """Count how many training runs are active across the cluster."""
        lock_dir = AI_SERVICE_ROOT / "data" / "coordination"
        if not lock_dir.exists():
            return 0

        count = 0
        for lock_file in lock_dir.glob("training.*.lock"):
            try:
                if lock_file.stat().st_size > 0:
                    age = time.time() - lock_file.stat().st_mtime
                    if age < 3600:
                        count += 1
            except (OSError, FileNotFoundError, PermissionError):
                continue
        return count

    def should_trigger_training(self) -> str | None:
        """Check if training should be triggered. Returns config key or None.

        Supports parallel training: different configs can train simultaneously,
        up to _max_concurrent_training total runs.
        """
        # Check global training_in_progress flag (legacy check for backwards compatibility)
        if self.state.training_in_progress:
            return None

        # Check concurrent training limit (cluster-wide)
        active_runs = self.count_active_training_runs()
        if active_runs >= self._max_concurrent_training:
            if self.config.verbose:
                print(f"[Training] Max concurrent training ({self._max_concurrent_training}) reached cluster-wide")
            return None

        # Duration-aware scheduling
        if HAS_COORDINATION:
            import socket

            from app.coordination.duration_scheduler import get_scheduler
            node_id = socket.gethostname()
            scheduler = get_scheduler()
            can_schedule, schedule_reason = scheduler.can_schedule_now(
                "training", node_id, avoid_peak_hours=False
            )
            if not can_schedule:
                if self.config.verbose:
                    print(f"[Training] Deferred by duration scheduler: {schedule_reason}")
                return None

        # Health-aware training
        if HAS_RESOURCE_OPTIMIZER and get_utilization_status is not None:
            try:
                util_status = get_utilization_status()
                gpu_util = util_status.get('gpu_util', 70)
                if gpu_util > 85:
                    return None
            except (AttributeError, KeyError, TypeError, ValueError):
                pass

        # Cluster health gate
        if HAS_PRE_SPAWN_HEALTH and gate_on_cluster_health is not None:
            try:
                can_proceed, health_msg = gate_on_cluster_health(
                    "training", min_healthy=2, min_healthy_ratio=0.4
                )
                if not can_proceed:
                    if self.config.verbose:
                        print(f"[Training] Deferred: {health_msg}")
                    return None
            except (AttributeError, TypeError, ValueError, ConnectionError):
                pass

        now = time.time()

        # Import ConfigPriorityQueue at runtime
        from scripts.unified_ai_loop import ConfigPriorityQueue
        priority_queue = ConfigPriorityQueue()
        prioritized_configs = priority_queue.get_prioritized_configs(self.state.configs)

        for config_key, _priority_score in prioritized_configs:
            config_state = self.state.configs[config_key]

            # Skip configs already training (per-config lock check)
            if self.is_training_locked_elsewhere(config_key):
                continue
            if config_key in self._training_locks:
                continue  # Already training locally

            if now - config_state.last_training_time < self.config.min_interval_seconds:
                continue

            # Momentum-based acceleration
            if HAS_FEEDBACK_ACCELERATOR:
                try:
                    decision = get_feedback_accelerator().get_training_decision(config_key)
                    if decision.should_train:
                        record_games_generated(config_key, config_state.games_since_training)
                        intensity_str = decision.intensity.value if decision.intensity else "normal"
                        momentum_str = decision.momentum.value if decision.momentum else "stable"
                        print(f"[Training] Trigger: momentum-based acceleration for {config_key} "
                              f"(intensity={intensity_str}, momentum={momentum_str})")
                        return config_key
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass

            # Dynamic game count threshold
            dynamic_threshold = self._get_dynamic_threshold(config_key)
            if config_state.games_since_training >= dynamic_threshold:
                print(f"[Training] Trigger: game threshold reached for {config_key} "
                      f"({config_state.games_since_training} >= {dynamic_threshold} games)")
                return config_key

            # Elo plateau detection
            if self.feedback and self.feedback.eval_analyzer.is_plateau(
                config_key,
                min_improvement=self.feedback_config.elo_plateau_threshold,
                lookback=self.feedback_config.elo_plateau_lookback
            ) and config_state.games_since_training >= self.config.trigger_threshold_games // 4:
                print(f"[Training] Trigger: Elo plateau detected for {config_key}")
                return config_key

            # Win rate degradation
            if self.feedback:
                weak_configs = self.feedback.eval_analyzer.get_weak_configs(
                    threshold=self.feedback_config.win_rate_degradation_threshold
                )
                if config_key in weak_configs and config_state.games_since_training >= self.config.trigger_threshold_games // 4:
                    print(f"[Training] Trigger: Win rate degradation for {config_key}")
                    return config_key

        return None

    async def start_training(self, config_key: str) -> bool:
        """Start a training run for the given configuration."""
        if DISABLE_LOCAL_TASKS:
            # Try remote dispatch via execution backend
            if self._backend is not None:
                return await self._dispatch_remote_training(config_key)
            print("[Training] Skipping local training (RINGRIFT_DISABLE_LOCAL_TASKS=true, no backend)")
            return False

        if self.state.training_in_progress:
            return False

        # Circuit breaker check - avoid spawning if too many recent failures (2025-12)
        if self._circuit_breaker and not self._circuit_breaker.can_execute("training_spawn"):
            print("[Training] BLOCKED by circuit breaker: training spawn circuit OPEN (too many recent failures)")
            if HAS_PROMETHEUS:
                DATA_QUALITY_BLOCKED_TRAINING.labels(reason="circuit_breaker_open").inc()
            return False

        self.record_training_start()

        # Data quality gate (enforced even without feedback controller if configured)
        enforce_gate = getattr(self.config, 'enforce_data_quality_gate', True)
        min_quality = getattr(self.config, 'min_data_quality_for_training', 0.7)

        if self.feedback:
            parity_failure_rate = self.feedback.data_monitor.get_parity_failure_rate()
            if parity_failure_rate > self.feedback_config.max_parity_failure_rate:
                print(f"[Training] BLOCKED by data quality gate: parity failure rate {parity_failure_rate:.1%}")
                if HAS_PROMETHEUS:
                    DATA_QUALITY_BLOCKED_TRAINING.labels(reason="parity_failure").inc()
                return False

            if self.feedback.should_quarantine_data():
                print("[Training] BLOCKED by data quality gate: data quarantined")
                if HAS_PROMETHEUS:
                    DATA_QUALITY_BLOCKED_TRAINING.labels(reason="quarantined").inc()
                return False

            if self.feedback.state.data_quality_score < self.feedback_config.min_data_quality_score:
                print(f"[Training] BLOCKED by data quality gate: score {self.feedback.state.data_quality_score:.2f}")
                if HAS_PROMETHEUS:
                    DATA_QUALITY_BLOCKED_TRAINING.labels(reason="low_score").inc()
                return False
        elif enforce_gate:
            # No feedback controller - use cached parity state if available
            if self._parity_failure_rate > getattr(self.config, 'parity_failure_threshold', 0.10):
                print(f"[Training] BLOCKED by data quality gate (no feedback): "
                      f"cached parity failure rate {self._parity_failure_rate:.1%}")
                if HAS_PROMETHEUS:
                    DATA_QUALITY_BLOCKED_TRAINING.labels(reason="parity_failure_cached").inc()
                return False

        parts = config_key.rsplit("_", 1)
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

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

            # Pre-training sync: fetch high-quality data from cluster (P5)
            synced_count = await self._sync_high_quality_data_before_training(config_key)
            if synced_count > 0:
                print(f"[Training] Pre-training sync added {synced_count} high-quality games")

            # NNUE dataset validation before training (2025-12)
            if HAS_NNUE_VALIDATION:
                db_path = self.config.selfplay_db_path / f"{config_key}.db"
                if db_path.exists():
                    try:
                        validation_result = validate_database_integrity(db_path, quick_check=True)
                        if not validation_result.is_valid:
                            print(f"[Training] WARNING: Database integrity check failed for {config_key}")
                            print(f"[Training]   Errors: {validation_result.errors[:3]}")  # First 3 errors
                            if HAS_PROMETHEUS:
                                VALIDATION_ERRORS.labels(error_type="db_integrity", config=config_key).inc()
                            # Don't block training, just warn (data may still be usable)
                        else:
                            print(f"[Training] Database integrity OK ({validation_result.samples_checked} samples checked)")
                    except Exception as e:
                        print(f"[Training] Database validation error (continuing anyway): {e}")

            if HAS_COORDINATION and estimate_task_duration is not None:
                est_duration = estimate_task_duration("training", config=config_key)  # type: ignore[misc]
                eta_time = datetime.fromtimestamp(time.time() + est_duration).strftime("%H:%M:%S")
                print(f"[Training] Estimated duration: {est_duration/3600:.1f}h (ETA: {eta_time})")

            model_version = "v3"

            games_dir = AI_SERVICE_ROOT / "data" / "games"
            synced_dir = games_dir / "synced"
            gpu_selfplay_dir = games_dir / "gpu_selfplay"
            selfplay_dir = AI_SERVICE_ROOT / "data" / "selfplay"

            # Check for JSONL data in multiple locations (local and synced)
            jsonl_paths = [
                gpu_selfplay_dir / config_key / "games.jsonl",  # Local GPU selfplay
                selfplay_dir / f"gpu_{config_key}" / "games.jsonl",  # Local selfplay
            ]
            # Also check synced selfplay directories from remote hosts
            synced_selfplay_dirs = [
                selfplay_dir / "gpu" / "synced",
                selfplay_dir / "p2p_gpu" / "synced",
                games_dir / "gpu_selfplay" / "synced",
            ]
            for sync_dir in synced_selfplay_dirs:
                if sync_dir.exists():
                    jsonl_paths.extend(sync_dir.rglob(f"*{config_key}*/*.jsonl"))
                    jsonl_paths.extend(sync_dir.rglob(f"*/{config_key}/*.jsonl"))

            # Filter to existing files with data
            jsonl_files = [p for p in jsonl_paths if p.exists() and p.stat().st_size > 0]
            has_jsonl_data = len(jsonl_files) > 0
            # Use first JSONL file as primary (for backward compatibility)
            jsonl_path = jsonl_files[0] if jsonl_files else None
            if has_jsonl_data:
                print(f"[Training] Found {len(jsonl_files)} JSONL files for {config_key}")

            game_dbs = list(games_dir.glob("*.db"))
            if synced_dir.exists():
                game_dbs.extend(synced_dir.rglob("*.db"))

            if not game_dbs and not has_jsonl_data:
                print("[Training] No game data found")
                self.state.training_in_progress = False
                self._release_training_lock()
                return False

            # Phase 2.5: Incremental DB consolidation
            # Track which DBs have been merged to avoid redundant work
            consolidated_db = games_dir / "consolidated_training_v2.db"
            consolidation_state_file = games_dir / ".consolidation_state.json"

            should_consolidate = False
            if has_jsonl_data:
                print("[Training] Using JSONL data, skipping DB consolidation")
            elif not consolidated_db.exists():
                should_consolidate = True
            else:
                # Check if any source DB has changed since last consolidation
                last_state = {}
                if consolidation_state_file.exists():
                    try:
                        with open(consolidation_state_file) as f:
                            last_state = json.load(f)
                    except (OSError, json.JSONDecodeError, PermissionError):
                        pass

                # Find DBs that need merging
                merge_dbs = []
                changed_dbs = []
                for db_path in game_dbs:
                    db_str = str(db_path)
                    if any(skip in db_str for skip in ['quarantine', 'corrupted', 'backup', 'consolidated']):
                        continue
                    if 'selfplay' in db_path.name or 'merged' in db_path.name or 'training' in db_path.name:
                        try:
                            current_mtime = db_path.stat().st_mtime
                            current_size = db_path.stat().st_size
                            db_key = str(db_path)
                            last_mtime = last_state.get(db_key, {}).get('mtime', 0)
                            last_size = last_state.get(db_key, {}).get('size', 0)

                            conn = sqlite3.connect(db_path)
                            cursor = conn.cursor()
                            cursor.execute("SELECT COUNT(*) FROM games")
                            count = cursor.fetchone()[0]
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'")
                            has_moves = cursor.fetchone() is not None
                            conn.close()

                            if count > 0 and has_moves:
                                merge_dbs.append(db_path)
                                # Check if this DB has new data
                                if current_mtime > last_mtime + 60 or current_size > last_size:
                                    changed_dbs.append(db_path.name)
                        except (sqlite3.Error, OSError, IndexError, TypeError):
                            pass

                # Only re-consolidate if there are changed DBs
                if changed_dbs:
                    should_consolidate = True
                    print(f"[Training] Incremental consolidation: {len(changed_dbs)} DBs with new data")

            if should_consolidate and not has_jsonl_data:
                # Re-scan for merge_dbs if we didn't already
                if 'merge_dbs' not in locals() or not merge_dbs:
                    merge_dbs = []
                    for db_path in game_dbs:
                        db_str = str(db_path)
                        if any(skip in db_str for skip in ['quarantine', 'corrupted', 'backup', 'consolidated']):
                            continue
                        if 'selfplay' in db_path.name or 'merged' in db_path.name or 'training' in db_path.name:
                            try:
                                conn = sqlite3.connect(db_path)
                                cursor = conn.cursor()
                                cursor.execute("SELECT COUNT(*) FROM games")
                                count = cursor.fetchone()[0]
                                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'")
                                has_moves = cursor.fetchone() is not None
                                conn.close()
                                if count > 0 and has_moves:
                                    merge_dbs.append(db_path)
                            except (sqlite3.Error, OSError, IndexError, TypeError):
                                pass

                if len(merge_dbs) > 1:
                    print(f"[Training] Consolidating {len(merge_dbs)} databases...")
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
                        await asyncio.wait_for(merge_process.communicate(), timeout=1800)

                        # Save consolidation state for incremental tracking
                        import json
                        new_state = {}
                        for db_path in merge_dbs:
                            with contextlib.suppress(Exception):
                                new_state[str(db_path)] = {
                                    'mtime': db_path.stat().st_mtime,
                                    'size': db_path.stat().st_size,
                                }
                        with open(consolidation_state_file, 'w') as f:
                            json.dump(new_state, f)
                        print(f"[Training] Consolidation state saved for {len(new_state)} DBs")
                    except Exception as e:
                        print(f"[Training] Consolidation error: {e}")

            if consolidated_db.exists():
                largest_db = consolidated_db
            else:
                largest_db = max(game_dbs, key=lambda p: p.stat().st_size)

            training_dir = AI_SERVICE_ROOT / "data" / "training"
            training_dir.mkdir(parents=True, exist_ok=True)
            data_path = training_dir / f"unified_{config_key}.npz"

            export_max_age = 6 * 3600
            skip_export = False
            if data_path.exists():
                npz_age = time.time() - data_path.stat().st_mtime
                if npz_age < export_max_age:
                    skip_export = True

            encoder_version = "v2" if board_type == "hexagonal" else "default"

            if not skip_export:
                if has_jsonl_data:
                    export_cmd = [
                        sys.executable,
                        str(AI_SERVICE_ROOT / "scripts" / "jsonl_to_npz.py"),
                        "--input", str(jsonl_path),
                        "--output", str(data_path),
                        "--board-type", board_type,
                        "--num-players", str(num_players),
                        "--gpu-selfplay",
                        "--max-games", "10000",
                    ]
                else:
                    export_cmd = [
                        sys.executable,
                        str(AI_SERVICE_ROOT / self.config.export_script),
                        "--db", str(largest_db),
                        "--output", str(data_path),
                        "--board-type", board_type,
                        "--num-players", str(num_players),
                        "--sample-every", "2",
                        "--require-completed",
                        "--min-moves", "10",
                        "--exclude-recovery",
                    ]
                if encoder_version != "default":
                    export_cmd.extend(["--encoder-version", encoder_version])

                # Feb 2026: Cross-process export coordination
                try:
                    from app.coordination.export_coordinator import get_export_coordinator
                    _coord = get_export_coordinator()
                    if not _coord.try_acquire(config_key):
                        logger.warning(f"Export slot unavailable for {config_key}, skipping training")
                        self.state.training_in_progress = False
                        self._release_training_lock()
                        return False
                    _release_export_slot = True
                except Exception:
                    _release_export_slot = False

                export_env = os.environ.copy()
                export_env["PYTHONPATH"] = str(AI_SERVICE_ROOT)
                try:
                    export_process = await asyncio.create_subprocess_exec(
                        *export_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=AI_SERVICE_ROOT,
                        env=export_env,
                    )
                    stdout, stderr = await export_process.communicate()
                finally:
                    if _release_export_slot:
                        try:
                            _coord.release(config_key)
                        except Exception:
                            pass

                if export_process.returncode != 0:
                    self.state.training_in_progress = False
                    self._release_training_lock()
                    return False

            timestamp = int(time.time())
            model_id = f"{config_key}_v3_{timestamp}"
            run_dir = AI_SERVICE_ROOT / "logs" / "unified_training" / model_id

            base_epochs = 100
            if self.feedback:
                epochs_multiplier = self.feedback.get_epochs_multiplier()
                epochs = max(50, int(base_epochs * epochs_multiplier))
            else:
                epochs = base_epochs

            # Get optimized training settings
            batch_size = self.config.batch_size or 256
            sampling_weights = self.config.sampling_weights or "victory_type"

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
                "--batch-size", str(batch_size),
                "--sampling-weights", sampling_weights,
                "--use-optimized-hyperparams",
                "--warmup-epochs", str(self.config.warmup_epochs),
            ]

            # Advanced training optimizations
            if self.config.use_spectral_norm:
                cmd.append("--spectral-norm")
            if self.config.use_lars:
                cmd.append("--lars")
            if self.config.use_cyclic_lr:
                cmd.extend(["--cyclic-lr", "--cyclic-lr-period", str(self.config.cyclic_lr_period)])
            if self.config.use_gradient_profiling:
                cmd.append("--gradient-profiling")
            if self.config.use_mixed_precision:
                cmd.extend(["--mixed-precision", "--amp-dtype", self.config.amp_dtype])
            if self.config.gradient_accumulation > 1:
                cmd.extend(["--gradient-accumulation", str(self.config.gradient_accumulation)])

            # 2025-12 Training Improvements: Policy label smoothing and hex augmentation
            # Policy label smoothing (prevents overconfident predictions)
            policy_smoothing = getattr(self.config, 'policy_label_smoothing', 0.05)
            if policy_smoothing > 0:
                cmd.extend(["--policy-label-smoothing", str(policy_smoothing)])

            # D6 hex symmetry augmentation for hex boards (12x effective data)
            if getattr(self.config, 'use_hex_augmentation', True) and board_type in ('hex8', 'hexagonal'):
                cmd.append("--augment-hex-symmetry")

            # Knowledge distillation
            if self.config.use_knowledge_distill and self.config.teacher_model_path:
                cmd.extend([
                    "--knowledge-distill",
                    "--teacher-model", self.config.teacher_model_path,
                    "--distill-alpha", str(self.config.distill_alpha),
                    "--distill-temperature", str(self.config.distill_temperature),
                ])


            # 2024-12 Advanced Training Improvements
            if getattr(self.config, 'use_value_whitening', True):
                cmd.extend([
                    "--value-whitening",
                    "--value-whitening-momentum", str(getattr(self.config, 'value_whitening_momentum', 0.99)),
                ])
            if getattr(self.config, 'use_ema', True):
                cmd.extend([
                    "--ema",
                    "--ema-decay", str(getattr(self.config, 'ema_decay', 0.999)),
                ])
            if getattr(self.config, 'use_stochastic_depth', True):
                cmd.extend([
                    "--stochastic-depth",
                    "--stochastic-depth-prob", str(getattr(self.config, 'stochastic_depth_prob', 0.1)),
                ])
            if getattr(self.config, 'use_adaptive_warmup', True):
                cmd.append("--adaptive-warmup")
            if getattr(self.config, 'use_hard_example_mining', True):
                cmd.extend([
                    "--hard-example-mining",
                    "--hard-example-top-k", str(getattr(self.config, 'hard_example_top_k', 0.3)),
                ])
            if getattr(self.config, 'use_dynamic_batch', False):
                cmd.extend([
                    "--dynamic-batch",
                    "--dynamic-batch-schedule", getattr(self.config, 'dynamic_batch_schedule', 'linear'),
                ])
            # Cross-board transfer learning
            if getattr(self.config, 'transfer_from_model', None):
                cmd.extend([
                    "--transfer-from", self.config.transfer_from_model,
                    "--transfer-freeze-epochs", str(getattr(self.config, 'transfer_freeze_epochs', 5)),
                ])
            # Advanced optimizer enhancements (2024-12)
            if getattr(self.config, 'use_lookahead', True):
                cmd.extend([
                    "--lookahead",
                    "--lookahead-k", str(getattr(self.config, 'lookahead_k', 5)),
                    "--lookahead-alpha", str(getattr(self.config, 'lookahead_alpha', 0.5)),
                ])
            if getattr(self.config, 'use_adaptive_clip', True):
                cmd.append("--adaptive-clip")
            if getattr(self.config, 'use_gradient_noise', False):
                cmd.extend([
                    "--gradient-noise",
                    "--gradient-noise-variance", str(getattr(self.config, 'gradient_noise_variance', 0.01)),
                ])
            if getattr(self.config, 'use_board_nas', True):
                cmd.append("--board-nas")
            if getattr(self.config, 'use_self_supervised', False):
                cmd.extend([
                    "--self-supervised",
                    "--ss-epochs", str(getattr(self.config, 'ss_epochs', 10)),
                    "--ss-projection-dim", str(getattr(self.config, 'ss_projection_dim', 128)),
                    "--ss-temperature", str(getattr(self.config, 'ss_temperature', 0.07)),
                ])
            if getattr(self.config, 'use_online_bootstrap', True):
                cmd.extend([
                    "--online-bootstrap",
                    "--bootstrap-temperature", str(getattr(self.config, 'bootstrap_temperature', 1.5)),
                    "--bootstrap-start-epoch", str(getattr(self.config, 'bootstrap_start_epoch', 10)),
                ])
            # Phase 2 Advanced Training Improvements (2024-12)
            if getattr(self.config, 'use_prefetch_gpu', True):
                cmd.append("--prefetch-gpu")
            if getattr(self.config, 'use_difficulty_curriculum', True):
                cmd.extend([
                    "--difficulty-curriculum",
                    "--curriculum-initial-threshold", str(getattr(self.config, 'curriculum_initial_threshold', 0.9)),
                    "--curriculum-final-threshold", str(getattr(self.config, 'curriculum_final_threshold', 0.3)),
                ])
            if getattr(self.config, 'use_quantized_eval', True):
                cmd.append("--quantized-eval")
            if getattr(self.config, 'use_attention', False):
                cmd.extend([
                    "--use-attention",
                    "--attention-heads", str(getattr(self.config, 'attention_heads', 4)),
                ])
            if getattr(self.config, 'use_moe', False):
                cmd.extend([
                    "--use-moe",
                    "--moe-experts", str(getattr(self.config, 'moe_experts', 4)),
                    "--moe-top-k", str(getattr(self.config, 'moe_top_k', 2)),
                ])
            if getattr(self.config, 'use_multitask', False):
                cmd.extend([
                    "--use-multitask",
                    "--multitask-weight", str(getattr(self.config, 'multitask_weight', 0.1)),
                ])
            if getattr(self.config, 'use_lamb', False):
                cmd.append("--use-lamb")
            if getattr(self.config, 'use_gradient_compression', False):
                cmd.extend([
                    "--gradient-compression",
                    "--compression-ratio", str(getattr(self.config, 'compression_ratio', 0.1)),
                ])
            if getattr(self.config, 'use_contrastive', False):
                cmd.extend([
                    "--contrastive-pretrain",
                    "--contrastive-weight", str(getattr(self.config, 'contrastive_weight', 0.1)),
                ])
            # Phase 3 Advanced Training Improvements (2024-12)
            if getattr(self.config, 'use_sam', False):
                cmd.extend([
                    "--use-sam",
                    "--sam-rho", str(getattr(self.config, 'sam_rho', 0.05)),
                ])
            if getattr(self.config, 'use_td_lambda', False):
                cmd.extend([
                    "--td-lambda",
                    "--td-lambda-value", str(getattr(self.config, 'td_lambda_value', 0.95)),
                ])
            if getattr(self.config, 'use_grokking_detection', True):
                cmd.append("--grokking-detection")
            if getattr(self.config, 'use_auxiliary_targets', False):
                cmd.extend([
                    "--auxiliary-targets",
                    "--auxiliary-weight", str(getattr(self.config, 'auxiliary_weight', 0.1)),
                ])
            if getattr(self.config, 'use_pruning', False):
                cmd.extend([
                    "--pruning",
                    "--pruning-ratio", str(getattr(self.config, 'pruning_ratio', 0.3)),
                ])
            if getattr(self.config, 'use_game_phase_network', False):
                cmd.append("--game-phase-network")
            if getattr(self.config, 'use_self_play', False):
                cmd.extend([
                    "--self-play",
                    "--self-play-buffer", str(getattr(self.config, 'self_play_buffer', 100000)),
                ])
            if getattr(self.config, 'use_distillation', False):
                teacher_path = getattr(self.config, 'distillation_teacher_path', None)
                if teacher_path:
                    cmd.extend([
                        "--distillation",
                        "--teacher-path", str(teacher_path),
                        "--distill-temp", str(getattr(self.config, 'distillation_temp', 4.0)),
                        "--distill-alpha", str(getattr(self.config, 'distillation_alpha', 0.7)),
                    ])

            # Phase 4: Training Stability & Acceleration (2024-12)
            # Note: These features are integrated but optional
            if getattr(self.config, 'use_adaptive_accumulation', False):
                cmd.append("--adaptive-accumulation")
            if getattr(self.config, 'use_activation_checkpointing', False):
                cmd.extend([
                    "--activation-checkpointing",
                    "--checkpoint-ratio", str(getattr(self.config, 'checkpoint_ratio', 0.5)),
                ])
            if getattr(self.config, 'use_flash_attention', False):
                cmd.append("--flash-attention")
            if getattr(self.config, 'use_dynamic_loss_scaling', False):
                cmd.append("--dynamic-loss-scaling")
            if getattr(self.config, 'use_elastic_training', False):
                cmd.append("--elastic-training")

            # Phase 5: Production Optimization (2024-12)
            if getattr(self.config, 'use_streaming_npz', False):
                cmd.extend([
                    "--streaming-npz",
                    "--streaming-chunk-size", str(getattr(self.config, 'streaming_chunk_size', 10000)),
                ])
            if getattr(self.config, 'use_profiling', False):
                cmd.append("--profile")
                profile_dir = getattr(self.config, 'profile_dir', None)
                if profile_dir:
                    cmd.extend(["--profile-dir", str(profile_dir)])

            # Integrated Training Enhancements (2025-12)
            # Master switch for unified enhancement system
            if getattr(self.config, 'use_integrated_enhancements', False):
                cmd.append("--use-integrated-enhancements")
                # Sub-features controlled by integrated manager
                if getattr(self.config, 'curriculum_enabled', True):
                    cmd.append("--enable-curriculum")
                if getattr(self.config, 'augmentation_enabled', True):
                    cmd.append("--enable-augmentation")
                if getattr(self.config, 'elo_weighting_enabled', True):
                    cmd.append("--enable-elo-weighting")

            # Fault Tolerance (2025-12)
            if not getattr(self.config, 'enable_circuit_breaker', True):
                cmd.append("--disable-circuit-breaker")
            if not getattr(self.config, 'enable_anomaly_detection', True):
                cmd.append("--disable-anomaly-detection")
            gradient_clip_mode = getattr(self.config, 'gradient_clip_mode', 'adaptive')
            cmd.extend(["--gradient-clip-mode", gradient_clip_mode])
            if gradient_clip_mode == 'fixed':
                cmd.extend([
                    "--gradient-clip-max-norm",
                    str(getattr(self.config, 'gradient_clip_max_norm', 1.0)),
                ])
            anomaly_spike = getattr(self.config, 'anomaly_spike_threshold', 3.0)
            if anomaly_spike != 3.0:  # Only add if non-default
                cmd.extend(["--anomaly-spike-threshold", str(anomaly_spike)])
            anomaly_grad = getattr(self.config, 'anomaly_gradient_threshold', 100.0)
            if anomaly_grad != 100.0:  # Only add if non-default
                cmd.extend(["--anomaly-gradient-threshold", str(anomaly_grad)])
            if not getattr(self.config, 'enable_graceful_shutdown', True):
                cmd.append("--disable-graceful-shutdown")

            print(f"[Training] Starting training for {model_id}...")

            # Also start NNUE policy training in parallel if configured
            nnue_policy_process = None
            if hasattr(self.config, 'nnue_policy_script'):
                nnue_policy_process = await self._start_nnue_policy_training(
                    board_type, num_players, largest_db, epochs, run_dir,
                    jsonl_path=jsonl_path if has_jsonl_data else None
                )
            self._training_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
            )

            # Record success with circuit breaker
            if self._circuit_breaker:
                self._circuit_breaker.record_success("training_spawn")

            return True

        except Exception as e:
            print(f"[TrainingScheduler] Error starting training: {e}")
            # Record failure with circuit breaker
            if self._circuit_breaker:
                self._circuit_breaker.record_failure("training_spawn")
            self.state.training_in_progress = False
            self._release_training_lock()
            return False

    async def _start_nnue_policy_training(
        self,
        board_type: str,
        num_players: int,
        db_path: Path,
        epochs: int,
        run_dir: Path,
        jsonl_path: Path | None = None,
    ) -> asyncio.subprocess.Process | None:
        """Start NNUE policy training with advanced optimizations.

        Uses the new training features: SWA, EMA, progressive batching,
        focal loss, auto-KL loss detection, and D6 hex augmentation.
        """
        try:
            nnue_run_dir = run_dir / "nnue_policy"
            nnue_run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(AI_SERVICE_ROOT / self.config.nnue_policy_script),
                "--db", str(db_path),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--epochs", str(epochs),
                "--run-dir", str(nnue_run_dir),
                "--batch-size", str(self.config.batch_size or 256),
                "--lr-scheduler", "cosine_warmup",
                "--grad-clip", "1.0",
            ]

            # Add JSONL source for MCTS policy data
            if jsonl_path and jsonl_path.exists():
                cmd.extend(["--jsonl", str(jsonl_path)])

            # Auto-KL loss detection (uses MCTS visit distributions when available)
            cmd.extend([
                "--auto-kl-loss",
                "--kl-min-coverage", "0.3",
                "--kl-min-samples", "50",
            ])

            # Mixed precision (AMP)
            if self.config.use_mixed_precision:
                cmd.append("--use-amp")

            # Stochastic Weight Averaging
            if getattr(self.config, 'use_swa', True):
                cmd.append("--use-swa")
                swa_start = int(epochs * getattr(self.config, 'swa_start_fraction', 0.75))
                cmd.extend(["--swa-start-epoch", str(swa_start)])

            # Exponential Moving Average
            if getattr(self.config, 'use_ema', True):
                cmd.append("--use-ema")
                cmd.extend(["--ema-decay", str(getattr(self.config, 'ema_decay', 0.999))])

            # Progressive batch sizing
            if getattr(self.config, 'use_progressive_batch', True):
                cmd.append("--progressive-batch")
                cmd.extend(["--min-batch-size", str(getattr(self.config, 'min_batch_size', 64))])
                cmd.extend(["--max-batch-size", str(getattr(self.config, 'max_batch_size', 512))])

            # Focal loss for hard sample mining
            focal_gamma = getattr(self.config, 'focal_gamma', 2.0)
            if focal_gamma > 0:
                cmd.extend(["--focal-gamma", str(focal_gamma)])

            # Label smoothing warmup
            warmup = getattr(self.config, 'label_smoothing_warmup', 5)
            if warmup > 0:
                cmd.extend(["--label-smoothing-warmup", str(warmup)])

            # D6 hex symmetry augmentation for hex boards (12x effective data)
            if getattr(self.config, 'use_hex_augmentation', True) and board_type in ('hex8', 'hexagonal'):
                cmd.append("--hex-augment")
                hex_augment_count = getattr(self.config, 'hex_augment_count', 6)
                cmd.extend(["--hex-augment-count", str(hex_augment_count)])

            # Policy head dropout for regularization
            policy_dropout = getattr(self.config, 'policy_dropout', 0.1)
            if policy_dropout > 0:
                cmd.extend(["--policy-dropout", str(policy_dropout)])

            # Gradient accumulation for larger effective batch sizes
            gradient_accumulation = getattr(self.config, 'gradient_accumulation', 1)
            if gradient_accumulation > 1:
                cmd.extend(["--gradient-accumulation-steps", str(gradient_accumulation)])

            # Learning rate finder (runs before training to find optimal LR)
            if getattr(self.config, 'use_lr_finder', False):
                cmd.append("--find-lr")
                lr_finder_iterations = getattr(self.config, 'lr_finder_iterations', 100)
                cmd.extend(["--lr-finder-iterations", str(lr_finder_iterations)])

            # Save learning curves
            cmd.append("--save-curves")

            jsonl_status = f"JSONL: {jsonl_path.name}" if jsonl_path and jsonl_path.exists() else "JSONL: None"
            print("[Training] Starting NNUE policy training with advanced optimizations...")
            print(f"[Training]   SWA: {getattr(self.config, 'use_swa', True)}, "
                  f"EMA: {getattr(self.config, 'use_ema', True)}, "
                  f"Progressive batch: {getattr(self.config, 'use_progressive_batch', True)}")
            print(f"[Training]   Focal gamma: {focal_gamma}, Policy dropout: {policy_dropout}, "
                  f"Hex augment: {getattr(self.config, 'use_hex_augmentation', True) and board_type in ('hex8', 'hexagonal')}")
            print(f"[Training]   Auto-KL: True (min_coverage=30%, min_samples=50), {jsonl_status}")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(AI_SERVICE_ROOT)

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
                env=env,
            )

            return process

        except Exception as e:
            print(f"[Training] Error starting NNUE policy training: {e}")
            return None

    async def check_training_status(self) -> dict[str, Any] | None:
        """Check if current training has completed."""
        if not self.state.training_in_progress or not self._training_process:
            return None

        if self._training_process.returncode is not None:
            _stdout, _stderr = await self._training_process.communicate()

            success = self._training_process.returncode == 0
            config_key = self.state.training_config

            self.state.training_in_progress = False
            self.state.training_config = ""
            self.state.total_training_runs += 1
            self._training_process = None
            self._release_training_lock()

            if config_key in self.state.configs:
                self.state.configs[config_key].last_training_time = time.time()
                self.state.configs[config_key].games_since_training = 0

            result = {
                "config": config_key,
                "success": success,
                "duration": time.time() - self.state.training_started_at,
            }

            if success and HAS_VALUE_CALIBRATION:
                calibration_report = await self._run_calibration_analysis(config_key)
                if calibration_report:
                    result["calibration"] = calibration_report

            if HAS_IMPROVEMENT_OPTIMIZER and success:
                try:
                    optimizer = get_improvement_optimizer()
                    calibration_ece = result.get("calibration", {}).get("ece")
                    optimizer.record_training_complete(
                        config_key=config_key,
                        duration_seconds=result["duration"],
                        val_loss=0.0,
                        calibration_ece=calibration_ece,
                    )
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass

            # PFSP Integration: Add successfully trained model to opponent pool
            if success and self._pfsp_pool is not None:
                try:
                    model_path = self._get_latest_model_path(config_key)
                    if model_path:
                        self._pfsp_pool.add_opponent(
                            model_id=model_path.stem,
                            model_path=str(model_path),
                            elo=INITIAL_ELO_RATING,  # Updated after evaluation
                            win_rate=0.5,
                        )
                        print(f"[PFSP] Added {model_path.stem} to opponent pool")
                except Exception as e:
                    print(f"[PFSP] Error adding model to pool: {e}")

            # CMA-ES Auto-Tuning: Check if optimization should trigger
            if success and config_key in self._cmaes_auto_tuners:
                try:
                    auto_tuner = self._cmaes_auto_tuners[config_key]
                    config_state = self.state.configs.get(config_key)
                    if config_state:
                        elo = config_state.current_elo or INITIAL_ELO_RATING
                        should_tune = auto_tuner.check_plateau(elo)
                        if should_tune:
                            print(f"[CMA-ES] Auto-tuning triggered for {config_key} (Elo plateau detected)")
                            asyncio.create_task(self._trigger_cmaes_auto_tuning(config_key))
                except Exception as e:
                    print(f"[CMA-ES] Error checking plateau: {e}")

            # Elo-based checkpoint selection: Find best checkpoint by playing strength
            if success and self._use_elo_checkpoint_selection:
                try:
                    model_path = self._get_latest_model_path(config_key)
                    if model_path:
                        candidate_id = model_path.stem
                        print(f"[Elo Selection] Triggering checkpoint evaluation for {candidate_id}")
                        asyncio.create_task(
                            self._run_elo_checkpoint_selection(config_key, candidate_id)
                        )
                except Exception as e:
                    print(f"[Elo Selection] Error triggering selection: {e}")

            # ELO Service Integration: Register new model and get training feedback
            if success and HAS_ELO_SERVICE:
                try:
                    elo_service = get_elo_service()
                    model_path = self._get_latest_model_path(config_key)
                    if model_path and elo_service:
                        parts = config_key.rsplit("_", 1)
                        board_type = parts[0]
                        num_players = int(parts[1].replace("p", "")) if len(parts) > 1 else 2

                        # Register newly trained model
                        elo_service.register_model(
                            model_id=model_path.stem,
                            board_type=board_type,
                            num_players=num_players,
                            model_path=str(model_path),
                        )
                        print(f"[ELO] Registered new model: {model_path.stem}")

                        # Get training feedback for next cycle
                        feedback = elo_service.get_training_feedback(board_type, num_players)
                        result["elo_feedback"] = {
                            "elo_stagnating": feedback.elo_stagnating,
                            "elo_regression": feedback.elo_regression,
                            "avg_elo_delta": feedback.avg_elo_delta,
                        }
                        if feedback.elo_stagnating:
                            print(f"[ELO] Warning: Elo stagnating for {config_key}")
                        if feedback.elo_regression:
                            print(f"[ELO] Warning: Elo regression detected for {config_key}")
                except Exception as e:
                    print(f"[ELO] Error registering model: {e}")

            # Update source quality based on training results (quality-aware feedback loop)
            if success and self._manifest:
                try:
                    await self._update_source_quality_after_training(config_key, result)
                except Exception as e:
                    print(f"[Training] Error updating source quality: {e}")

            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.TRAINING_COMPLETED,
                payload=result
            ))

            return result

        return None

    async def _run_calibration_analysis(self, config_key: str) -> dict[str, Any] | None:
        """Run value calibration analysis on recent training data."""
        if not HAS_VALUE_CALIBRATION:
            return None

        try:
            if config_key not in self._calibration_trackers:
                self._calibration_trackers[config_key] = CalibrationTracker(window_size=1000)

            tracker = self._calibration_trackers[config_key]
            report = tracker.compute_current_calibration()
            if report is None:
                return None

            report_dict = report.to_dict()
            self.state.calibration_reports[config_key] = report_dict
            self.state.last_calibration_time = time.time()

            print(f"[Calibration] {config_key}: ECE={report.ece:.4f}, MCE={report.mce:.4f}")

            return report_dict

        except Exception as e:
            print(f"[Calibration] Error analyzing {config_key}: {e}")
            return None

    def _get_latest_model_path(self, config_key: str) -> Path | None:
        """Get the path to the latest trained model for a config."""
        try:
            models_dir = AI_SERVICE_ROOT / "models"
            pattern = f"{config_key}*.pth"
            models = list(models_dir.glob(pattern))
            if not models:
                # Try .pt extension
                pattern = f"{config_key}*.pt"
                models = list(models_dir.glob(pattern))
            if models:
                return max(models, key=lambda p: p.stat().st_mtime)
            return None
        except (OSError, ValueError, AttributeError):
            return None

    async def _trigger_cmaes_auto_tuning(self, config_key: str) -> None:
        """Trigger CMA-ES auto-tuning for a specific config."""
        try:
            parts = config_key.rsplit("_", 1)
            board_type = parts[0]
            num_players = int(parts[1].replace("p", "")) if len(parts) > 1 else 2

            # Find CMA-ES tuning script
            cmaes_script = AI_SERVICE_ROOT / "scripts" / "hyperparameter_tuning.py"
            if not cmaes_script.exists():
                cmaes_script = AI_SERVICE_ROOT / "scripts" / "cmaes_hp_search.py"

            if not cmaes_script.exists():
                print("[CMA-ES] Tuning script not found")
                return

            cmd = [
                sys.executable,
                str(cmaes_script),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--generations", "50",
                "--population-size", "16",
                "--games-per-eval", "20",
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = str(AI_SERVICE_ROOT)

            print(f"[CMA-ES] Starting auto-tuning for {config_key}")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=AI_SERVICE_ROOT,
                env=env,
            )

            # Run in background, don't wait
            asyncio.create_task(self._monitor_cmaes_process(config_key, process))

        except Exception as e:
            print(f"[CMA-ES] Error triggering auto-tuning: {e}")

    async def _monitor_cmaes_process(self, config_key: str, process) -> None:
        """Monitor CMA-ES process completion."""
        try:
            _stdout, stderr = await process.communicate()
            if process.returncode == 0:
                print(f"[CMA-ES] Auto-tuning completed for {config_key}")
                # Mark auto-tuner as completed
                if config_key in self._cmaes_auto_tuners:
                    self._cmaes_auto_tuners[config_key].mark_tuning_complete()
            else:
                print(f"[CMA-ES] Auto-tuning failed for {config_key}: {stderr.decode()[:200]}")
        except Exception as e:
            print(f"[CMA-ES] Error monitoring process: {e}")

    async def _run_elo_checkpoint_selection(
        self,
        config_key: str,
        candidate_id: str,
    ) -> None:
        """Run Elo-based checkpoint selection in a thread pool.

        This evaluates all checkpoints for a training run by playing games
        against baseline opponents, selecting the best by playing strength
        rather than validation loss.
        """
        if not HAS_ELO_CHECKPOINT_SELECTION or select_best_checkpoint is None:
            return

        try:
            # Parse board type from config key (e.g., "square8_2p" -> BoardType.SQUARE8)
            parts = config_key.rsplit("_", 1)
            board_type_str = parts[0]
            num_players = int(parts[1].replace("p", "")) if len(parts) > 1 else 2

            # Import BoardType here to avoid circular imports
            from app.models import BoardType
            board_type_map = {
                "square8": BoardType.SQUARE8,
                "square19": BoardType.SQUARE19,
                "hexagonal": BoardType.HEXAGONAL,
                "hex8": BoardType.HEX8,
            }
            board_type = board_type_map.get(board_type_str)
            if board_type is None:
                print(f"[Elo Selection] Unknown board type: {board_type_str}")
                return

            print(f"[Elo Selection] Evaluating checkpoints for {candidate_id}...")

            # Run CPU-bound evaluation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            best_ckpt, results = await loop.run_in_executor(
                None,  # Use default thread pool
                lambda: select_best_checkpoint(
                    candidate_id=candidate_id,
                    models_dir=str(AI_SERVICE_ROOT / "models"),
                    games_per_opponent=self._elo_selection_games_per_opponent,
                    board_type=board_type,
                    num_players=num_players,
                ),
            )

            if best_ckpt:
                # Find Elo for best checkpoint
                best_elo = None
                for r in results:
                    if r["checkpoint"] == str(best_ckpt):
                        best_elo = r.get("estimated_elo")
                        break

                print(f"[Elo Selection] Best checkpoint: {best_ckpt.name} (Elo: {best_elo:.0f})")

                # Optionally copy best to models/<candidate_id>_elo_best.pth
                if self._elo_selection_copy_best:
                    import shutil
                    best_path = AI_SERVICE_ROOT / "models" / f"{candidate_id}_elo_best.pth"
                    shutil.copy2(best_ckpt, best_path)
                    print(f"[Elo Selection] Copied to: {best_path}")

                # Update PFSP pool with accurate Elo if available
                if self._pfsp_pool is not None and best_elo is not None:
                    with contextlib.suppress(Exception):
                        self._pfsp_pool.update_stats(
                            best_ckpt.stem,
                            elo=best_elo,
                        )

            else:
                print(f"[Elo Selection] No valid checkpoints found for {candidate_id}")

        except Exception as e:
            print(f"[Elo Selection] Error during evaluation: {e}")
            import traceback
            traceback.print_exc()

    def get_temperature_for_move(self, move_number: int, game_state: Any | None = None) -> float:
        """Get exploration temperature for a given move in self-play."""
        if self._temp_scheduler is None:
            return 1.0
        return self._temp_scheduler.get_temperature(move_number, game_state)

    def update_training_progress(self, progress: float):
        """Update training progress for curriculum-based temperature scheduling."""
        if self._temp_scheduler is not None:
            self._temp_scheduler.set_training_progress(progress)

    async def request_urgent_training(self, configs: list[str], reason: str) -> bool:
        """Request urgent training for specified configs due to feedback signal."""
        if self.state.training_in_progress:
            print("[Training] Urgent training request deferred: training already in progress")
            return False

        now = time.time()
        for config_key in configs:
            if config_key not in self.state.configs:
                continue

            config_state = self.state.configs[config_key]
            urgent_cooldown = self.config.min_interval_seconds / 2
            if now - config_state.last_training_time < urgent_cooldown:
                continue

            print(f"[Training] URGENT TRAINING triggered for {config_key}: {reason}")
            started = await self.start_training(config_key)
            if started:
                return True

        print(f"[Training] Urgent training request could not be fulfilled for configs: {configs}")
        return False

    # =========================================================================
    # Advanced Training Utilities (2025-12)
    # =========================================================================

    def get_pfsp_opponent(self, config_key: str, current_elo: float = INITIAL_ELO_RATING) -> str | None:
        """Get an opponent from PFSP pool for self-play.

        Args:
            config_key: Config identifier for filtering
            current_elo: Current model Elo for opponent selection

        Returns:
            Model path of selected opponent or None if pool empty
        """
        if self._pfsp_pool is None:
            return None

        opponent = self._pfsp_pool.sample_opponent(
            current_elo=current_elo,
            strategy="pfsp",
        )
        return opponent.model_path if opponent else None

    def add_pfsp_opponent(
        self,
        model_path: str,
        elo: float = INITIAL_ELO_RATING,
        generation: int = 0,
        name: str | None = None,
    ) -> None:
        """Add an opponent to the PFSP pool.

        Args:
            model_path: Path to model file
            elo: Model's Elo rating
            generation: Training generation number
            name: Optional display name
        """
        if self._pfsp_pool is not None:
            self._pfsp_pool.add_opponent(model_path, elo, generation, name)

    def update_pfsp_stats(
        self,
        model_path: str,
        won: bool,
        drew: bool = False,
        elo_change: float = 0.0,
    ) -> None:
        """Update PFSP opponent statistics after a game.

        Args:
            model_path: Opponent model path
            won: Whether current model won
            drew: Whether game was a draw
            elo_change: Elo rating change
        """
        if self._pfsp_pool is not None:
            self._pfsp_pool.update_stats(model_path, won, drew, elo_change)

    def get_pfsp_pool_stats(self) -> dict[str, Any]:
        """Get PFSP opponent pool statistics."""
        if self._pfsp_pool is not None:
            return self._pfsp_pool.get_pool_stats()
        return {'size': 0}

    def update_cmaes_metrics(
        self,
        config_key: str,
        current_elo: float | None = None,
        current_loss: float | None = None,
        current_win_rate: float | None = None,
    ) -> None:
        """Update CMA-ES auto-tuner with current training metrics.

        Args:
            config_key: Config identifier
            current_elo: Current Elo rating
            current_loss: Current training loss
            current_win_rate: Current win rate
        """
        if hasattr(self, '_cmaes_auto_tuners') and config_key in self._cmaes_auto_tuners:
            self._cmaes_auto_tuners[config_key].step(
                current_elo=current_elo,
                current_loss=current_loss,
                current_win_rate=current_win_rate,
            )

    def should_trigger_cmaes(self, config_key: str) -> bool:
        """Check if CMA-ES auto-tuning should be triggered for a config.

        Returns:
            True if plateau detected and tuning should run
        """
        if hasattr(self, '_cmaes_auto_tuners') and config_key in self._cmaes_auto_tuners:
            return self._cmaes_auto_tuners[config_key].should_tune()
        return False

    async def run_cmaes_auto_tuning(self, config_key: str) -> dict[str, Any] | None:
        """Run CMA-ES hyperparameter optimization for a config.

        Args:
            config_key: Config identifier

        Returns:
            Optimization result with weights, or None if failed
        """
        if not hasattr(self, '_cmaes_auto_tuners') or config_key not in self._cmaes_auto_tuners:
            return None

        tuner = self._cmaes_auto_tuners[config_key]
        print(f"[Training] Starting CMA-ES auto-tuning for {config_key}")

        # Run optimization (this is blocking but can take hours)
        result = tuner.run_optimization(
            generations=getattr(self.config, 'cmaes_generations', 30),
            population_size=getattr(self.config, 'cmaes_population_size', 15),
        )

        if result:
            print(f"[Training] CMA-ES optimization complete for {config_key}")
            await self.event_bus.publish(DataEvent(
                event_type=DataEventType.TRAINING_COMPLETED,
                payload={
                    'type': 'cmaes_auto_tune',
                    'config': config_key,
                    'result': result,
                }
            ))

        return result

    def get_cmaes_status(self, config_key: str) -> dict[str, Any]:
        """Get CMA-ES auto-tuner status for a config."""
        if hasattr(self, '_cmaes_auto_tuners') and config_key in self._cmaes_auto_tuners:
            return self._cmaes_auto_tuners[config_key].get_status()
        return {'available': False}

    def enable_gradient_checkpointing(self, model: Any) -> bool:
        """Enable gradient checkpointing for memory-efficient training.

        Args:
            model: PyTorch model to apply checkpointing to

        Returns:
            True if successfully enabled
        """
        if not HAS_ADVANCED_TRAINING or GradientCheckpointing is None:
            return False

        try:
            layers = getattr(self.config, 'gradient_checkpoint_layers', None)
            self._gradient_checkpointing = GradientCheckpointing(model, layers)
            self._gradient_checkpointing.enable()
            print("[Training] Gradient checkpointing enabled")
            return True
        except Exception as e:
            print(f"[Training] Failed to enable gradient checkpointing: {e}")
            return False

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        if self._gradient_checkpointing is not None:
            self._gradient_checkpointing.disable()
            self._gradient_checkpointing = None
            print("[Training] Gradient checkpointing disabled")

    # =========================================================================
    # Bottleneck Fix Integration Methods (2025-12)
    # =========================================================================

    async def start_streaming_pipelines(self) -> None:
        """Start streaming data pipelines for all configs.

        This enables real-time data ingestion with async DB polling,
        eliminating blocking consolidation waits.
        """
        for config_key, pipeline in self._streaming_pipelines.items():
            try:
                await pipeline.start()
                print(f"[Training] Streaming pipeline started for {config_key}")
            except Exception as e:
                print(f"[Training] Failed to start streaming for {config_key}: {e}")

    async def stop_streaming_pipelines(self) -> None:
        """Stop all streaming data pipelines."""
        for _config_key, pipeline in self._streaming_pipelines.items():
            with contextlib.suppress(Exception):
                await pipeline.stop()

    def get_streaming_stats(self, config_key: str | None = None) -> dict[str, Any]:
        """Get streaming pipeline statistics.

        Args:
            config_key: Optional config to get stats for (None for all)

        Returns:
            Dict with streaming stats including buffer sizes and ingestion counts
        """
        if config_key and config_key in self._streaming_pipelines:
            return self._streaming_pipelines[config_key].get_stats()

        return {
            key: pipeline.get_stats()
            for key, pipeline in self._streaming_pipelines.items()
        }

    def record_parity_failure(self, config_key: str, passed: bool) -> None:
        """Record a parity validation result for training decision feedback.

        High parity failure rates indicate GPU/CPU divergence and should
        trigger more conservative training thresholds.

        Args:
            config_key: Config identifier
            passed: True if validation passed
        """
        # Update rolling parity failure rate (exponential moving average)
        alpha = 0.1  # Smoothing factor
        result = 0.0 if passed else 1.0
        self._parity_failure_rate = alpha * result + (1 - alpha) * self._parity_failure_rate

        # Also record to async validator if available
        if self._async_validator and not passed:
            # Log divergence for debugging
            print(f"[Training] Parity failure recorded for {config_key}, "
                  f"rolling rate: {self._parity_failure_rate:.2%}")

    def get_parity_failure_rate(self) -> float:
        """Get current parity failure rate.

        Returns:
            Rolling average parity failure rate (0.0-1.0)
        """
        return self._parity_failure_rate

    def should_block_training_for_parity(self, threshold: float = 0.10) -> bool:
        """Check if training should be blocked due to high parity failures.

        Args:
            threshold: Maximum acceptable failure rate (default 10%)

        Returns:
            True if training should be blocked
        """
        if self._parity_failure_rate > threshold:
            print(f"[Training] BLOCKING: Parity failure rate {self._parity_failure_rate:.2%} > {threshold:.0%}")
            if HAS_PROMETHEUS and DATA_QUALITY_BLOCKED_TRAINING:
                DATA_QUALITY_BLOCKED_TRAINING.labels(reason='parity_failure').inc()
            return True
        return False

    def get_async_validator_report(self) -> dict[str, Any]:
        """Get async shadow validation report.

        Returns:
            Validation stats including divergence rates and job counts
        """
        if self._async_validator is not None:
            return self._async_validator.get_report()
        return {'enabled': False}

    def check_validation_error(self) -> bool:
        """Check if async validator has triggered an error threshold.

        Returns:
            True if validation error threshold was exceeded
        """
        if self._async_validator is not None:
            return self._async_validator.check_error()
        return False

    def get_connection_pool_stats(self) -> dict[str, Any]:
        """Get connection pool statistics.

        Returns:
            Stats including connections created, reused, and reuse ratio
        """
        if self._connection_pool is not None:
            return self._connection_pool.get_stats()
        return {'enabled': False}

    def get_bottleneck_fix_status(self) -> dict[str, Any]:
        """Get comprehensive status of all bottleneck fix integrations.

        Returns:
            Dict with status of streaming, validation, and connection pooling
        """
        return {
            'streaming_pipelines': {
                'enabled': len(self._streaming_pipelines) > 0,
                'count': len(self._streaming_pipelines),
                'stats': self.get_streaming_stats(),
            },
            'async_validation': {
                'enabled': self._async_validator is not None,
                'report': self.get_async_validator_report(),
            },
            'connection_pool': {
                'enabled': self._connection_pool is not None,
                'stats': self.get_connection_pool_stats(),
            },
            'parity_failure_rate': self._parity_failure_rate,
        }

    # =========================================================================
    # Consolidated Feedback State Management (2025-12)
    # =========================================================================

    def sync_feedback_state(self, config_key: str) -> None:
        """Sync global signals to per-config FeedbackState.

        Propagates global parity failure rate and curriculum weights
        to the consolidated FeedbackState for the given config.

        Args:
            config_key: Config identifier (e.g., "square8_2p")
        """
        config_state = self.state.configs.get(config_key)
        if config_state is None:
            return

        # Ensure feedback state exists
        if not hasattr(config_state, 'feedback') or config_state.feedback is None:
            config_state.feedback = FeedbackState()

        feedback = config_state.feedback

        # Sync global parity failure rate
        feedback.parity_failure_rate = self._parity_failure_rate

        # Sync curriculum weight from scheduler's tracking
        feedback.curriculum_weight = self.get_curriculum_weight(config_key)
        feedback.curriculum_last_update = time.time()

        # Sync elo from config state (keep in sync)
        feedback.elo_current = config_state.current_elo
        feedback.elo_trend = config_state.elo_trend

        # Sync win rate from config state
        feedback.win_rate = config_state.win_rate
        feedback.win_rate_trend = config_state.win_rate_trend
        feedback.consecutive_high_win_rate = config_state.consecutive_high_win_rate

        # Recompute urgency
        feedback.compute_urgency()

    def get_config_feedback(self, config_key: str) -> FeedbackState | None:
        """Get consolidated FeedbackState for a config.

        Args:
            config_key: Config identifier

        Returns:
            FeedbackState or None if config doesn't exist
        """
        config_state = self.state.configs.get(config_key)
        if config_state is None:
            return None

        # Sync before returning to ensure latest data
        self.sync_feedback_state(config_key)
        return config_state.feedback

    def update_config_feedback(
        self,
        config_key: str,
        elo: float | None = None,
        win_rate: float | None = None,
        parity_passed: bool | None = None,
        curriculum_weight: float | None = None,
    ) -> None:
        """Update feedback signals for a config.

        Args:
            config_key: Config identifier
            elo: New Elo rating (triggers plateau detection)
            win_rate: New win rate (triggers streak tracking)
            parity_passed: Parity check result (updates failure rate)
            curriculum_weight: New curriculum weight
        """
        config_state = self.state.configs.get(config_key)
        if config_state is None:
            return

        # Ensure feedback state exists
        if not hasattr(config_state, 'feedback') or config_state.feedback is None:
            config_state.feedback = FeedbackState()

        feedback = config_state.feedback

        # Update Elo with plateau detection
        if elo is not None:
            feedback.update_elo(elo)
            # Sync back to config state
            config_state.current_elo = elo
            config_state.elo_trend = feedback.elo_trend

        # Update win rate with streak tracking
        if win_rate is not None:
            feedback.update_win_rate(win_rate)
            # Sync back to config state
            config_state.win_rate = win_rate
            config_state.win_rate_trend = feedback.win_rate_trend
            config_state.consecutive_high_win_rate = feedback.consecutive_high_win_rate

        # Update parity failure rate
        if parity_passed is not None:
            feedback.update_parity(parity_passed)
            # Also update global parity rate
            self.record_parity_failure(config_key, parity_passed)

        # Update curriculum weight
        if curriculum_weight is not None:
            feedback.curriculum_weight = curriculum_weight
            feedback.curriculum_last_update = time.time()
            self._curriculum_weights[config_key] = curriculum_weight

        # Recompute urgency after updates
        feedback.compute_urgency()

    def get_all_feedback_states(self) -> dict[str, dict[str, Any]]:
        """Get consolidated feedback states for all configs.

        Returns:
            Dict mapping config_key to FeedbackState as dict
        """
        result = {}
        for config_key in self.state.configs:
            feedback = self.get_config_feedback(config_key)
            if feedback is not None:
                result[config_key] = feedback.to_dict()
        return result

    def get_most_urgent_config(self) -> str | None:
        """Get the config with highest training urgency.

        Uses the consolidated FeedbackState urgency score to
        determine which config most needs training.

        Returns:
            Config key with highest urgency, or None if no configs
        """
        if not self.state.configs:
            return None

        best_config = None
        best_urgency = -1.0

        for config_key in self.state.configs:
            feedback = self.get_config_feedback(config_key)
            if feedback is not None and feedback.urgency_score > best_urgency:
                best_urgency = feedback.urgency_score
                best_config = config_key

        if best_config and best_urgency > 0.3:
            print(f"[Training] Most urgent config: {best_config} (urgency={best_urgency:.2f})")

        return best_config

    def should_trigger_cmaes_for_config(self, config_key: str) -> bool:
        """Check if CMA-ES auto-tuning should trigger for a config.

        Based on consecutive Elo plateaus tracked in FeedbackState.

        Args:
            config_key: Config identifier

        Returns:
            True if CMA-ES should trigger
        """
        feedback = self.get_config_feedback(config_key)
        if feedback is None:
            return False

        plateau_threshold = getattr(self.feedback_config, 'plateau_count_for_cmaes', 2)
        return feedback.elo_plateau_count >= plateau_threshold

    def get_feedback_summary(self) -> dict[str, Any]:
        """Get summary of all feedback signals across configs.

        Returns:
            Summary dict with averages and per-config details
        """
        all_feedback = self.get_all_feedback_states()

        if not all_feedback:
            return {'configs': 0, 'avg_urgency': 0.0, 'avg_parity_failure': 0.0}

        urgencies = [f['urgency_score'] for f in all_feedback.values()]
        parity_rates = [f['parity_failure_rate'] for f in all_feedback.values()]
        plateau_counts = [f['elo_plateau_count'] for f in all_feedback.values()]

        return {
            'configs': len(all_feedback),
            'avg_urgency': sum(urgencies) / len(urgencies),
            'max_urgency': max(urgencies),
            'avg_parity_failure': sum(parity_rates) / len(parity_rates),
            'global_parity_failure': self._parity_failure_rate,
            'total_plateau_count': sum(plateau_counts),
            'configs_in_plateau': sum(1 for p in plateau_counts if p >= 2),
            'per_config': all_feedback,
        }

    # =========================================================================
    # Training Auto-Recovery (Phase 7)
    # =========================================================================

    def schedule_training_retry(self, config_key: str) -> float | None:
        """Schedule a retry for a failed training run.

        Uses exponential backoff with jitter based on retry count. Returns the
        scheduled retry time, or None if max retries exceeded.

        Jitter is applied to prevent thundering herd when multiple training
        runs fail simultaneously (e.g., cluster-wide issue).

        Args:
            config_key: Config that failed training

        Returns:
            Scheduled retry timestamp, or None if max retries exceeded
        """
        # Use RetryPolicy if available, otherwise fall back to config values
        if self._retry_policy and HAS_RETRY_POLICY:
            max_retries = self._retry_policy.get('max_retries', 3)
            base_delay = self._retry_policy.get('base_delay', 60.0)
            multiplier = self._retry_policy.get('exponential_base', 2.0)
            use_jitter = self._retry_policy.get('jitter', True)
        else:
            max_retries = getattr(self.config, 'training_max_retries', 3)
            base_delay = getattr(self.config, 'training_retry_backoff_base', 60.0)
            multiplier = getattr(self.config, 'training_retry_backoff_multiplier', 2.0)
            use_jitter = True  # Default to jitter for safety

        current_retries = self._retry_attempts.get(config_key, 0)

        if current_retries >= max_retries:
            print(f"[Training] Max retries ({max_retries}) exceeded for {config_key}")
            # Reset retry count for future attempts
            self._retry_attempts[config_key] = 0
            return None

        # Increment retry count
        self._retry_attempts[config_key] = current_retries + 1
        self._last_failure_time[config_key] = time.time()

        # Emit retry metric
        if HAS_PROMETHEUS and TRAINING_RETRY_ATTEMPTS:
            TRAINING_RETRY_ATTEMPTS.labels(config=config_key).inc()

        # Calculate delay with exponential backoff
        delay = base_delay * (multiplier ** current_retries)

        # Add jitter to prevent thundering herd (2025-12)
        if use_jitter:
            import random
            # Add +/- 25% jitter
            jitter_range = delay * 0.25
            delay = delay + random.uniform(-jitter_range, jitter_range)
            delay = max(base_delay, delay)  # Ensure minimum delay

        scheduled_time = time.time() + delay

        # Add to pending retries
        self._pending_retries.append((config_key, scheduled_time))
        self._pending_retries.sort(key=lambda x: x[1])  # Sort by scheduled time

        print(f"[Training] Scheduled retry {current_retries + 1}/{max_retries} for {config_key} "
              f"in {delay:.0f}s (at {datetime.fromtimestamp(scheduled_time).strftime('%H:%M:%S')})"
              f"{' [+jitter]' if use_jitter else ''}")

        return scheduled_time

    def reset_retry_count(self, config_key: str) -> None:
        """Reset retry count after successful training.

        Args:
            config_key: Config that completed successfully
        """
        if config_key in self._retry_attempts:
            old_count = self._retry_attempts[config_key]
            del self._retry_attempts[config_key]
            if old_count > 0:
                print(f"[Training] Reset retry count for {config_key} (was {old_count})")
                # Emit success after retry metric
                if HAS_PROMETHEUS and TRAINING_RETRY_SUCCESS:
                    TRAINING_RETRY_SUCCESS.labels(config=config_key, attempt_number=str(old_count)).inc()

        if config_key in self._last_failure_time:
            del self._last_failure_time[config_key]

        # Remove from pending retries
        self._pending_retries = [
            (k, t) for k, t in self._pending_retries if k != config_key
        ]

    def get_pending_retry(self) -> str | None:
        """Get a config that is due for retry.

        Returns:
            Config key that should be retried, or None if none due
        """
        if not self._pending_retries:
            return None

        now = time.time()
        # Check first pending retry (list is sorted by time)
        config_key, scheduled_time = self._pending_retries[0]

        if now >= scheduled_time:
            # Remove from pending list
            self._pending_retries.pop(0)
            return config_key

        return None

    async def start_training_with_retry(self, config_key: str) -> bool:
        """Start training with automatic retry on failure.

        Wraps start_training with retry logic. On failure, schedules
        a retry with exponential backoff.

        Args:
            config_key: Config to train

        Returns:
            True if training started successfully
        """
        try:
            success = await self.start_training(config_key)

            if success:
                # Training started - reset retry count
                self.reset_retry_count(config_key)
                return True
            else:
                # Training failed to start - schedule retry
                self.schedule_training_retry(config_key)
                return False

        except Exception as e:
            print(f"[Training] Exception during training start: {e}")
            self.schedule_training_retry(config_key)
            return False

    def get_retry_status(self) -> dict[str, Any]:
        """Get current retry status for all configs.

        Returns:
            Dict with retry counts and pending retries
        """
        return {
            'retry_counts': dict(self._retry_attempts),
            'last_failures': {
                k: datetime.fromtimestamp(v).isoformat()
                for k, v in self._last_failure_time.items()
            },
            'pending_retries': [
                {'config': k, 'scheduled_at': datetime.fromtimestamp(t).isoformat()}
                for k, t in self._pending_retries
            ],
            'configs_at_max_retries': [
                k for k, v in self._retry_attempts.items()
                if v >= getattr(self.config, 'training_max_retries', 3)
            ],
        }

    # =========================================================================
    # Post-Promotion Warmup (Phase 7)
    # =========================================================================

    def is_in_warmup_period(self, config_key: str) -> tuple[bool, str]:
        """Check if config is in post-promotion warmup period.

        After a model is promoted, we wait for:
        1. Minimum time to pass (default 30 min)
        2. Minimum games to be collected (default 100)

        This allows the new model to generate diverse training data
        before we train again.

        Args:
            config_key: Config to check

        Returns:
            Tuple of (is_in_warmup, reason_string)
        """
        config_state = self.state.configs.get(config_key)
        if config_state is None:
            return False, "config_not_found"

        warmup_games = getattr(self.config, 'warmup_games_after_promotion', 100)
        warmup_time = getattr(self.config, 'warmup_time_after_promotion', 1800.0)

        now = time.time()
        time_since_promotion = now - config_state.last_promotion_time
        games_since = config_state.games_since_training

        # Check time warmup
        if time_since_promotion < warmup_time:
            remaining = warmup_time - time_since_promotion
            return True, f"time_warmup: {remaining:.0f}s remaining"

        # Check games warmup
        if games_since < warmup_games:
            remaining = warmup_games - games_since
            return True, f"games_warmup: {remaining} games remaining"

        return False, "warmup_complete"

    def get_warmup_status(self) -> dict[str, dict[str, Any]]:
        """Get warmup status for all configs.

        Returns:
            Dict mapping config_key to warmup status
        """
        result = {}
        for config_key, config_state in self.state.configs.items():
            in_warmup, reason = self.is_in_warmup_period(config_key)
            result[config_key] = {
                'in_warmup': in_warmup,
                'reason': reason,
                'time_since_promotion': time.time() - config_state.last_promotion_time,
                'games_since_training': config_state.games_since_training,
            }
        return result

    # =========================================================================
    # Training Checkpointing (Phase 7)
    # =========================================================================

    def _get_checkpoint_path(self) -> Path:
        """Get the path to the checkpoint file."""
        custom_path = getattr(self.config, 'checkpoint_path', None)
        if custom_path:
            return Path(custom_path)
        return AI_SERVICE_ROOT / "data" / "training_checkpoint.json"

    def save_checkpoint(self) -> bool:
        """Save current training state to checkpoint file.

        Saves:
        - Retry attempts and pending retries
        - Curriculum weights
        - Feedback state summary
        - Last checkpoint time

        Returns:
            True if checkpoint saved successfully
        """
        if not getattr(self.config, 'checkpoint_enabled', True):
            return False

        try:
            import json
            checkpoint_path = self._get_checkpoint_path()
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint_data = {
                'timestamp': time.time(),
                'timestamp_iso': datetime.now().isoformat(),
                'retry_attempts': self._retry_attempts,
                'last_failure_times': self._last_failure_time,
                'pending_retries': [
                    {'config': k, 'scheduled_time': t}
                    for k, t in self._pending_retries
                ],
                'curriculum_weights': self._curriculum_weights,
                'parity_failure_rate': self._parity_failure_rate,
                'promotion_history': list(self._promotion_history),
                'training_history': list(self._training_history),
                'config_states': {
                    k: {
                        'games_since_training': cs.games_since_training,
                        'last_training_time': cs.last_training_time,
                        'last_promotion_time': cs.last_promotion_time,
                        'current_elo': cs.current_elo,
                        'win_rate': cs.win_rate,
                    }
                    for k, cs in self.state.configs.items()
                },
            }

            # Atomic write with temp file
            temp_path = checkpoint_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            temp_path.rename(checkpoint_path)

            if self.config.verbose:
                print(f"[Checkpoint] Saved to {checkpoint_path}")
            return True

        except Exception as e:
            print(f"[Checkpoint] Error saving checkpoint: {e}")
            return False

    def load_checkpoint(self) -> bool:
        """Load training state from checkpoint file.

        Restores:
        - Retry attempts and pending retries
        - Curriculum weights
        - Parity failure rate

        Returns:
            True if checkpoint loaded successfully
        """
        try:
            import json
            checkpoint_path = self._get_checkpoint_path()

            if not checkpoint_path.exists():
                print(f"[Checkpoint] No checkpoint file found at {checkpoint_path}")
                return False

            with open(checkpoint_path) as f:
                checkpoint_data = json.load(f)

            # Check if checkpoint is recent (within 24 hours)
            checkpoint_time = checkpoint_data.get('timestamp', 0)
            age_hours = (time.time() - checkpoint_time) / 3600
            if age_hours > 24:
                print(f"[Checkpoint] Checkpoint is {age_hours:.1f}h old, skipping restore")
                return False

            # Restore retry state
            self._retry_attempts = checkpoint_data.get('retry_attempts', {})
            self._last_failure_time = checkpoint_data.get('last_failure_times', {})
            self._pending_retries = [
                (r['config'], r['scheduled_time'])
                for r in checkpoint_data.get('pending_retries', [])
            ]

            # Restore curriculum weights
            self._curriculum_weights = checkpoint_data.get('curriculum_weights', {})

            # Restore parity failure rate
            self._parity_failure_rate = checkpoint_data.get('parity_failure_rate', 0.0)

            # Restore history deques (limited to recent entries)
            promotion_history = checkpoint_data.get('promotion_history', [])
            self._promotion_history = deque(promotion_history[-100:], maxlen=100)
            training_history = checkpoint_data.get('training_history', [])
            self._training_history = deque(training_history[-100:], maxlen=100)

            print(f"[Checkpoint] Restored from {checkpoint_path} "
                  f"(age: {age_hours:.1f}h, retries: {len(self._retry_attempts)}, "
                  f"curriculum weights: {len(self._curriculum_weights)})")
            return True

        except Exception as e:
            print(f"[Checkpoint] Error loading checkpoint: {e}")
            return False

    def maybe_save_checkpoint(self, force: bool = False) -> bool:
        """Save checkpoint if enough time has elapsed.

        Args:
            force: Save regardless of interval

        Returns:
            True if checkpoint was saved
        """
        if not getattr(self.config, 'checkpoint_enabled', True):
            return False

        checkpoint_interval = getattr(self.config, 'checkpoint_interval_seconds', 300.0)

        if not hasattr(self, '_last_checkpoint_time'):
            self._last_checkpoint_time = 0.0

        now = time.time()
        if (force or (now - self._last_checkpoint_time) >= checkpoint_interval) and self.save_checkpoint():
            self._last_checkpoint_time = now
            return True
        return False

    # =========================================================================
    # A/B Testing for Hyperparameters (Phase 7)
    # =========================================================================

    def is_ab_test_config(self, config_key: str) -> bool:
        """Check if a config should use test hyperparameters.

        Uses deterministic selection based on config_key hash to ensure
        consistent assignment across restarts.

        Args:
            config_key: Config identifier

        Returns:
            True if config should use test hyperparameters
        """
        if not getattr(self.config, 'ab_testing_enabled', False):
            return False

        ab_fraction = getattr(self.config, 'ab_test_fraction', 0.3)

        # Deterministic selection based on config_key hash
        config_hash = hash(config_key) % 1000
        threshold = int(ab_fraction * 1000)

        return config_hash < threshold

    def get_ab_test_assignments(self) -> dict[str, str]:
        """Get A/B test group assignments for all configs.

        Returns:
            Dict mapping config_key to 'control' or 'test'
        """
        return {
            config_key: 'test' if self.is_ab_test_config(config_key) else 'control'
            for config_key in self.state.configs
        }

    def get_hyperparameters_for_config(self, config_key: str) -> dict[str, Any]:
        """Get hyperparameters for a config, applying A/B test overrides.

        Args:
            config_key: Config identifier

        Returns:
            Dict of hyperparameters to use
        """
        # Base hyperparameters from config
        base_params = {
            'batch_size': getattr(self.config, 'batch_size', 256),
            'epochs': 50,
            'learning_rate': 1e-3,
            'use_swa': getattr(self.config, 'use_swa', True),
            'use_ema': getattr(self.config, 'use_ema', True),
        }

        if self.is_ab_test_config(config_key):
            # Test group: Use experimental hyperparameters
            test_params = {
                'batch_size': 512,  # Larger batch size
                'learning_rate': 5e-4,  # Adjusted for larger batch
                'use_sam': True,  # Sharpness-Aware Minimization
                'use_lookahead': True,  # Lookahead optimizer
            }
            base_params.update(test_params)
            print(f"[A/B Test] {config_key} using TEST hyperparameters")
        else:
            print(f"[A/B Test] {config_key} using CONTROL hyperparameters")

        return base_params
