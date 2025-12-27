"""Unified Event Type Definitions - Single source of truth for all RingRift events.

This module consolidates all event type enums from:
- app.distributed.data_events.DataEventType (~140 types)
- app.coordination.stage_events.StageEvent (22 types)
- Cross-process event patterns

December 2025: Created for Phase 2 consolidation to eliminate duplicate definitions.

Usage:
    from app.events.types import RingRiftEventType, EventCategory

    # Use unified event types
    event_type = RingRiftEventType.TRAINING_COMPLETED

    # Check event category
    category = EventCategory.from_event(event_type)

    # For backwards compatibility, aliases are provided:
    from app.events.types import DataEventType, StageEvent
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class EventCategory(Enum):
    """Categories for organizing events."""

    DATA = "data"  # Data collection, sync, freshness
    TRAINING = "training"  # Training lifecycle
    EVALUATION = "evaluation"  # Model evaluation
    PROMOTION = "promotion"  # Model promotion
    CURRICULUM = "curriculum"  # Curriculum management
    SELFPLAY = "selfplay"  # Selfplay operations
    OPTIMIZATION = "optimization"  # CMA-ES, NAS, PBT
    QUALITY = "quality"  # Data quality
    REGRESSION = "regression"  # Regression detection
    CLUSTER = "cluster"  # Cluster/P2P operations
    SYSTEM = "system"  # Daemons, health, resources
    WORK = "work"  # Work queue
    STAGE = "stage"  # Pipeline stage completion
    SYNC = "sync"  # Synchronization and locking
    TASK = "task"  # Task lifecycle

    @classmethod
    def from_event(cls, event_type: RingRiftEventType) -> EventCategory:
        """Get the category for an event type."""
        return _EVENT_CATEGORIES.get(event_type, EventCategory.SYSTEM)


class RingRiftEventType(Enum):
    """Unified event types for all RingRift systems.

    This enum consolidates all event types from data_events.DataEventType
    and stage_events.StageEvent into a single source of truth.

    Organized by category for easy navigation.
    """

    # =========================================================================
    # DATA COLLECTION EVENTS
    # =========================================================================
    NEW_GAMES_AVAILABLE = "new_games"
    """New games have been generated and are ready for sync."""

    DATA_SYNC_STARTED = "sync_started"
    """Data synchronization has started."""

    DATA_SYNC_COMPLETED = "sync_completed"
    """Data synchronization completed successfully."""

    DATA_SYNC_FAILED = "sync_failed"
    """Data synchronization failed."""

    GAME_SYNCED = "game_synced"
    """Individual game(s) synced to targets."""

    DATABASE_CREATED = "database_created"
    """New database file created - immediate registration needed."""

    ORPHAN_GAMES_DETECTED = "orphan_games_detected"
    """Unregistered game databases found."""

    ORPHAN_GAMES_REGISTERED = "orphan_games_registered"
    """Orphan databases auto-registered."""

    # =========================================================================
    # DATA FRESHNESS EVENTS
    # =========================================================================
    DATA_STALE = "data_stale"
    """Training data is stale and needs refresh."""

    DATA_FRESH = "data_fresh"
    """Training data is fresh and ready for use."""

    SYNC_TRIGGERED = "sync_triggered"
    """Sync triggered due to stale data."""

    SYNC_REQUEST = "sync_request"
    """Explicit sync request (router-driven)."""

    SYNC_STALLED = "sync_stalled"
    """Sync operation stalled/timed out."""

    # =========================================================================
    # TRAINING EVENTS
    # =========================================================================
    TRAINING_THRESHOLD_REACHED = "training_threshold"
    """Sufficient data accumulated to start training."""

    TRAINING_STARTED = "training_started"
    """Training has started."""

    TRAINING_PROGRESS = "training_progress"
    """Training progress update (epoch completed)."""

    TRAINING_COMPLETED = "training_completed"
    """Training completed successfully."""

    TRAINING_FAILED = "training_failed"
    """Training failed with error."""

    TRAINING_LOSS_ANOMALY = "training_loss_anomaly"
    """Training loss spike detected."""

    TRAINING_LOSS_TREND = "training_loss_trend"
    """Training loss trend (improving/stalled/degrading)."""

    TRAINING_EARLY_STOPPED = "training_early_stopped"
    """Early stopping triggered (stagnation/regression)."""

    TRAINING_ROLLBACK_NEEDED = "training_rollback_needed"
    """Rollback to previous checkpoint recommended."""

    TRAINING_ROLLBACK_COMPLETED = "training_rollback_completed"
    """Training rollback completed."""

    # =========================================================================
    # EVALUATION EVENTS
    # =========================================================================
    EVALUATION_STARTED = "evaluation_started"
    """Model evaluation has started."""

    EVALUATION_PROGRESS = "evaluation_progress"
    """Evaluation progress update."""

    EVALUATION_COMPLETED = "evaluation_completed"
    """Model evaluation completed."""

    EVALUATION_FAILED = "evaluation_failed"
    """Model evaluation failed."""

    ELO_UPDATED = "elo_updated"
    """Elo rating updated for a model."""

    ELO_SIGNIFICANT_CHANGE = "elo_significant_change"
    """Elo change exceeded threshold - triggers curriculum rebalance."""

    ELO_VELOCITY_CHANGED = "elo_velocity_changed"
    """Elo improvement velocity changed significantly."""

    ADAPTIVE_PARAMS_CHANGED = "adaptive_params_changed"
    """Training params adjusted based on Elo momentum."""

    # =========================================================================
    # PROMOTION EVENTS
    # =========================================================================
    PROMOTION_CANDIDATE = "promotion_candidate"
    """Model is a candidate for promotion."""

    PROMOTION_STARTED = "promotion_started"
    """Promotion process started."""

    MODEL_PROMOTED = "model_promoted"
    """Model successfully promoted to production."""

    PROMOTION_FAILED = "promotion_failed"
    """Promotion failed with error."""

    PROMOTION_REJECTED = "promotion_rejected"
    """Promotion rejected (insufficient improvement)."""

    PROMOTION_ROLLED_BACK = "promotion_rolled_back"
    """Promotion rolled back due to regression."""

    MODEL_UPDATED = "model_updated"
    """Model metadata or path updated (pre-promotion)."""

    # =========================================================================
    # CURRICULUM EVENTS
    # =========================================================================
    CURRICULUM_REBALANCED = "curriculum_rebalanced"
    """Curriculum weights rebalanced."""

    CURRICULUM_ADVANCED = "curriculum_advanced"
    """Moved to harder curriculum tier."""

    WEIGHT_UPDATED = "weight_updated"
    """Curriculum/opponent weight updated."""

    OPPONENT_MASTERED = "opponent_mastered"
    """Opponent mastered - advance curriculum."""

    # =========================================================================
    # SELFPLAY EVENTS
    # =========================================================================
    SELFPLAY_COMPLETE = "selfplay_complete"
    """Selfplay batch finished."""

    SELFPLAY_TARGET_UPDATED = "selfplay_target_updated"
    """Request more/fewer selfplay games."""

    SELFPLAY_RATE_CHANGED = "selfplay_rate_changed"
    """Selfplay rate multiplier changed (>20%)."""

    IDLE_RESOURCE_DETECTED = "idle_resource_detected"
    """Idle GPU/CPU detected - can be used for selfplay."""

    # =========================================================================
    # OPTIMIZATION EVENTS (CMA-ES, NAS, PBT)
    # =========================================================================
    CMAES_TRIGGERED = "cmaes_triggered"
    """CMA-ES hyperparameter optimization triggered."""

    CMAES_COMPLETED = "cmaes_completed"
    """CMA-ES optimization completed."""

    NAS_TRIGGERED = "nas_triggered"
    """Neural Architecture Search triggered."""

    NAS_STARTED = "nas_started"
    """NAS started."""

    NAS_GENERATION_COMPLETE = "nas_generation_complete"
    """NAS generation completed."""

    NAS_COMPLETED = "nas_completed"
    """NAS completed."""

    NAS_BEST_ARCHITECTURE = "nas_best_architecture"
    """Best architecture found by NAS."""

    PBT_STARTED = "pbt_started"
    """Population-Based Training started."""

    PBT_GENERATION_COMPLETE = "pbt_generation_complete"
    """PBT generation completed."""

    PBT_COMPLETED = "pbt_completed"
    """PBT completed."""

    PLATEAU_DETECTED = "plateau_detected"
    """Training plateau detected - may trigger optimization."""

    HYPERPARAMETER_UPDATED = "hyperparameter_updated"
    """Hyperparameter updated (manual or automated)."""

    # =========================================================================
    # PRIORITIZED EXPERIENCE REPLAY EVENTS
    # =========================================================================
    PER_BUFFER_REBUILT = "per_buffer_rebuilt"
    """PER buffer rebuilt with new priorities."""

    PER_PRIORITIES_UPDATED = "per_priorities_updated"
    """PER priorities updated for samples."""

    # =========================================================================
    # TIER/GATING EVENTS
    # =========================================================================
    TIER_PROMOTION = "tier_promotion"
    """Difficulty tier promotion."""

    CROSSBOARD_PROMOTION = "crossboard_promotion"
    """Multi-config promotion decision."""

    # =========================================================================
    # PARITY VALIDATION EVENTS
    # =========================================================================
    PARITY_VALIDATION_STARTED = "parity_validation_started"
    """TS/Python parity validation started."""

    PARITY_VALIDATION_COMPLETED = "parity_validation_completed"
    """Parity validation completed."""

    PARITY_FAILURE_RATE_CHANGED = "parity_failure_rate_changed"
    """Parity failure rate changed significantly."""

    # =========================================================================
    # DATA QUALITY EVENTS
    # =========================================================================
    DATA_QUALITY_ALERT = "data_quality_alert"
    """General data quality alert."""

    QUALITY_CHECK_REQUESTED = "quality_check_requested"
    """On-demand quality check requested."""

    QUALITY_CHECK_FAILED = "quality_check_failed"
    """Quality check failed."""

    QUALITY_SCORE_UPDATED = "quality_score_updated"
    """Game quality score recalculated."""

    QUALITY_DISTRIBUTION_CHANGED = "quality_distribution_changed"
    """Significant shift in quality distribution."""

    HIGH_QUALITY_DATA_AVAILABLE = "high_quality_data_available"
    """Ready for training with high-quality data."""

    QUALITY_DEGRADED = "quality_degraded"
    """Quality dropped below threshold."""

    LOW_QUALITY_DATA_WARNING = "low_quality_data_warning"
    """Data quality below warning threshold."""

    TRAINING_BLOCKED_BY_QUALITY = "training_blocked_by_quality"
    """Quality too low to train."""

    QUALITY_FEEDBACK_ADJUSTED = "quality_feedback_adjusted"
    """Quality feedback updated for config."""

    SCHEDULER_REGISTERED = "scheduler_registered"
    """Temperature scheduler registered for config."""

    QUALITY_PENALTY_APPLIED = "quality_penalty_applied"
    """Penalty applied - reduce selfplay rate."""

    EXPLORATION_BOOST = "exploration_boost"
    """Request to boost exploration temperature."""

    EXPLORATION_ADJUSTED = "exploration_adjusted"
    """Exploration strategy changed."""

    # =========================================================================
    # REGISTRY & METRICS EVENTS
    # =========================================================================
    REGISTRY_UPDATED = "registry_updated"
    """Model/data registry updated."""

    METRICS_UPDATED = "metrics_updated"
    """Metrics dashboard updated."""

    CACHE_INVALIDATED = "cache_invalidated"
    """Cache invalidated - reload required."""

    # =========================================================================
    # REGRESSION DETECTION EVENTS
    # =========================================================================
    REGRESSION_DETECTED = "regression_detected"
    """Any regression detected."""

    REGRESSION_MINOR = "regression_minor"
    """Minor regression (severity: minor)."""

    REGRESSION_MODERATE = "regression_moderate"
    """Moderate regression."""

    REGRESSION_SEVERE = "regression_severe"
    """Severe regression."""

    REGRESSION_CRITICAL = "regression_critical"
    """Critical regression - rollback recommended."""

    REGRESSION_CLEARED = "regression_cleared"
    """Model recovered from regression."""

    # =========================================================================
    # P2P/CLUSTER EVENTS
    # =========================================================================
    P2P_MODEL_SYNCED = "p2p_model_synced"
    """Model synced across P2P cluster."""

    MODEL_SYNC_REQUESTED = "model_sync_requested"
    """Model sync requested."""

    P2P_CLUSTER_HEALTHY = "p2p_cluster_healthy"
    """P2P cluster is healthy."""

    P2P_CLUSTER_UNHEALTHY = "p2p_cluster_unhealthy"
    """P2P cluster is unhealthy."""

    P2P_NODE_DEAD = "p2p_node_dead"
    """Single P2P node detected as dead (Dec 2025)."""

    P2P_NODES_DEAD = "p2p_nodes_dead"
    """P2P nodes detected as dead (batch)."""

    P2P_SELFPLAY_SCALED = "p2p_selfplay_scaled"
    """P2P selfplay scaled up/down."""

    CLUSTER_STATUS_CHANGED = "cluster_status_changed"
    """Cluster status changed."""

    CLUSTER_CAPACITY_CHANGED = "cluster_capacity_changed"
    """Cluster capacity changed."""

    NODE_UNHEALTHY = "node_unhealthy"
    """Node detected as unhealthy."""

    NODE_RECOVERED = "node_recovered"
    """Node recovered to healthy state."""

    NODE_ACTIVATED = "node_activated"
    """Node activated by cluster activator."""

    NODE_CAPACITY_UPDATED = "node_capacity_updated"
    """Node capacity metrics updated."""

    NODE_OVERLOADED = "node_overloaded"
    """Node CPU/GPU utilization critical."""

    # =========================================================================
    # SYSTEM/DAEMON EVENTS
    # =========================================================================
    DAEMON_STARTED = "daemon_started"
    """Daemon process started."""

    DAEMON_STOPPED = "daemon_stopped"
    """Daemon process stopped."""

    DAEMON_STATUS_CHANGED = "daemon_status_changed"
    """Daemon health status changed (stuck, crashed, restarted)."""

    HOST_ONLINE = "host_online"
    """Host came online."""

    HOST_OFFLINE = "host_offline"
    """Host went offline."""

    ERROR = "error"
    """General error event."""

    # =========================================================================
    # HEALTH & RECOVERY EVENTS
    # =========================================================================
    HEALTH_CHECK_PASSED = "health_check_passed"
    """Health check passed."""

    HEALTH_CHECK_FAILED = "health_check_failed"
    """Health check failed."""

    HEALTH_ALERT = "health_alert"
    """General health warning."""

    RESOURCE_CONSTRAINT = "resource_constraint"
    """CPU/GPU/Memory/Disk pressure."""

    RESOURCE_CONSTRAINT_DETECTED = "resource_constraint_detected"
    """Resource limit hit."""

    RECOVERY_INITIATED = "recovery_initiated"
    """Auto-recovery started."""

    RECOVERY_COMPLETED = "recovery_completed"
    """Auto-recovery finished."""

    RECOVERY_FAILED = "recovery_failed"
    """Auto-recovery failed."""

    MODEL_CORRUPTED = "model_corrupted"
    """Model file corruption detected."""

    COORDINATOR_HEALTHY = "coordinator_healthy"
    """Coordinator healthy signal."""

    COORDINATOR_UNHEALTHY = "coordinator_unhealthy"
    """Coordinator unhealthy signal."""

    COORDINATOR_HEALTH_DEGRADED = "coordinator_health_degraded"
    """Coordinator not fully healthy."""

    COORDINATOR_SHUTDOWN = "coordinator_shutdown"
    """Graceful coordinator shutdown."""

    COORDINATOR_INIT_FAILED = "coordinator_init_failed"
    """Coordinator failed to initialize."""

    COORDINATOR_HEARTBEAT = "coordinator_heartbeat"
    """Liveness signal from coordinator."""

    HANDLER_TIMEOUT = "handler_timeout"
    """Event handler timed out."""

    HANDLER_FAILED = "handler_failed"
    """Event handler threw exception."""

    # =========================================================================
    # WORK QUEUE EVENTS
    # =========================================================================
    WORK_QUEUED = "work_queued"
    """New work added to queue."""

    WORK_CLAIMED = "work_claimed"
    """Work claimed by a node."""

    WORK_STARTED = "work_started"
    """Work execution started."""

    WORK_COMPLETED = "work_completed"
    """Work completed successfully."""

    WORK_FAILED = "work_failed"
    """Work failed permanently."""

    WORK_RETRY = "work_retry"
    """Work failed, will retry."""

    WORK_TIMEOUT = "work_timeout"
    """Work timed out."""

    WORK_CANCELLED = "work_cancelled"
    """Work cancelled."""

    JOB_PREEMPTED = "job_preempted"
    """Job preempted for higher priority work."""

    # =========================================================================
    # LOCK/SYNCHRONIZATION EVENTS
    # =========================================================================
    LOCK_ACQUIRED = "lock_acquired"
    """Distributed lock acquired."""

    LOCK_RELEASED = "lock_released"
    """Distributed lock released."""

    LOCK_TIMEOUT = "lock_timeout"
    """Lock acquisition timed out."""

    DEADLOCK_DETECTED = "deadlock_detected"
    """Deadlock detected in distributed system."""

    # =========================================================================
    # CHECKPOINT EVENTS
    # =========================================================================
    CHECKPOINT_SAVED = "checkpoint_saved"
    """Training checkpoint saved."""

    CHECKPOINT_LOADED = "checkpoint_loaded"
    """Training checkpoint loaded."""

    # =========================================================================
    # TASK LIFECYCLE EVENTS
    # =========================================================================
    TASK_SPAWNED = "task_spawned"
    """Task spawned on a node."""

    TASK_HEARTBEAT = "task_heartbeat"
    """Task heartbeat (liveness signal)."""

    TASK_COMPLETED = "task_completed"
    """Task completed successfully."""

    TASK_FAILED = "task_failed"
    """Task failed with error."""

    TASK_ORPHANED = "task_orphaned"
    """Task orphaned (stopped sending heartbeats)."""

    TASK_CANCELLED = "task_cancelled"
    """Task cancelled by system."""

    TASK_ABANDONED = "task_abandoned"
    """Task intentionally abandoned (not orphaned)."""

    # =========================================================================
    # CAPACITY/BACKPRESSURE EVENTS
    # =========================================================================
    BACKPRESSURE_ACTIVATED = "backpressure_activated"
    """Backpressure activated (queue full)."""

    BACKPRESSURE_RELEASED = "backpressure_released"
    """Backpressure released (queue cleared)."""

    # =========================================================================
    # LEADER ELECTION EVENTS
    # =========================================================================
    LEADER_ELECTED = "leader_elected"
    """New cluster leader elected."""

    LEADER_LOST = "leader_lost"
    """Cluster leader lost."""

    LEADER_STEPDOWN = "leader_stepdown"
    """Leader stepping down gracefully."""

    # =========================================================================
    # ENCODING/PROCESSING EVENTS
    # =========================================================================
    ENCODING_BATCH_COMPLETED = "encoding_batch_completed"
    """Batch encoding of games completed."""

    CALIBRATION_COMPLETED = "calibration_completed"
    """Difficulty calibration completed."""

    # =========================================================================
    # STAGE COMPLETION EVENTS (from StageEvent)
    # =========================================================================
    # These represent pipeline stage completions with STAGE_ prefix
    # to distinguish from general events like TRAINING_COMPLETED

    STAGE_SELFPLAY_COMPLETE = "selfplay_complete"
    """Selfplay pipeline stage completed."""

    STAGE_CANONICAL_SELFPLAY_COMPLETE = "canonical_selfplay_complete"
    """Canonical selfplay pipeline stage completed."""

    STAGE_GPU_SELFPLAY_COMPLETE = "gpu_selfplay_complete"
    """GPU selfplay pipeline stage completed."""

    STAGE_SYNC_COMPLETE = "sync_complete"
    """Sync pipeline stage completed."""

    STAGE_PARITY_VALIDATION_COMPLETE = "parity_validation_complete"
    """Parity validation pipeline stage completed."""

    STAGE_NPZ_EXPORT_COMPLETE = "npz_export_complete"
    """NPZ export pipeline stage completed."""

    STAGE_TRAINING_COMPLETE = "stage_training_complete"
    """Training pipeline stage completed (distinct from TRAINING_COMPLETED)."""

    STAGE_TRAINING_STARTED = "stage_training_started"
    """Training pipeline stage started (distinct from TRAINING_STARTED)."""

    STAGE_TRAINING_FAILED = "stage_training_failed"
    """Training pipeline stage failed (distinct from TRAINING_FAILED)."""

    STAGE_EVALUATION_COMPLETE = "stage_evaluation_complete"
    """Evaluation pipeline stage completed."""

    STAGE_SHADOW_TOURNAMENT_COMPLETE = "shadow_tournament_complete"
    """Shadow tournament pipeline stage completed."""

    STAGE_ELO_CALIBRATION_COMPLETE = "elo_calibration_complete"
    """Elo calibration pipeline stage completed."""

    STAGE_CMAES_COMPLETE = "stage_cmaes_complete"
    """CMA-ES pipeline stage completed."""

    STAGE_PBT_COMPLETE = "stage_pbt_complete"
    """PBT pipeline stage completed."""

    STAGE_NAS_COMPLETE = "stage_nas_complete"
    """NAS pipeline stage completed."""

    STAGE_PROMOTION_COMPLETE = "stage_promotion_complete"
    """Promotion pipeline stage completed."""

    STAGE_TIER_GATING_COMPLETE = "tier_gating_complete"
    """Tier gating pipeline stage completed."""

    STAGE_ITERATION_COMPLETE = "iteration_complete"
    """Full training iteration completed."""

    STAGE_CLUSTER_SYNC_COMPLETE = "cluster_sync_complete"
    """Cluster sync pipeline stage completed."""

    STAGE_MODEL_SYNC_COMPLETE = "model_sync_complete"
    """Model sync pipeline stage completed."""


# =============================================================================
# Event Category Mapping
# =============================================================================

_EVENT_CATEGORIES: dict[RingRiftEventType, EventCategory] = {
    # Data events
    RingRiftEventType.NEW_GAMES_AVAILABLE: EventCategory.DATA,
    RingRiftEventType.DATA_SYNC_STARTED: EventCategory.DATA,
    RingRiftEventType.DATA_SYNC_COMPLETED: EventCategory.DATA,
    RingRiftEventType.DATA_SYNC_FAILED: EventCategory.DATA,
    RingRiftEventType.GAME_SYNCED: EventCategory.DATA,
    RingRiftEventType.DATA_STALE: EventCategory.DATA,
    RingRiftEventType.DATA_FRESH: EventCategory.DATA,
    RingRiftEventType.SYNC_TRIGGERED: EventCategory.DATA,
    RingRiftEventType.SYNC_REQUEST: EventCategory.DATA,
    RingRiftEventType.SYNC_STALLED: EventCategory.DATA,
    RingRiftEventType.DATABASE_CREATED: EventCategory.DATA,
    RingRiftEventType.ORPHAN_GAMES_DETECTED: EventCategory.DATA,
    RingRiftEventType.ORPHAN_GAMES_REGISTERED: EventCategory.DATA,

    # Training events
    RingRiftEventType.TRAINING_THRESHOLD_REACHED: EventCategory.TRAINING,
    RingRiftEventType.TRAINING_STARTED: EventCategory.TRAINING,
    RingRiftEventType.TRAINING_PROGRESS: EventCategory.TRAINING,
    RingRiftEventType.TRAINING_COMPLETED: EventCategory.TRAINING,
    RingRiftEventType.TRAINING_FAILED: EventCategory.TRAINING,
    RingRiftEventType.TRAINING_LOSS_ANOMALY: EventCategory.TRAINING,
    RingRiftEventType.TRAINING_LOSS_TREND: EventCategory.TRAINING,
    RingRiftEventType.TRAINING_EARLY_STOPPED: EventCategory.TRAINING,
    RingRiftEventType.TRAINING_ROLLBACK_NEEDED: EventCategory.TRAINING,
    RingRiftEventType.TRAINING_ROLLBACK_COMPLETED: EventCategory.TRAINING,

    # Evaluation events
    RingRiftEventType.EVALUATION_STARTED: EventCategory.EVALUATION,
    RingRiftEventType.EVALUATION_PROGRESS: EventCategory.EVALUATION,
    RingRiftEventType.EVALUATION_COMPLETED: EventCategory.EVALUATION,
    RingRiftEventType.EVALUATION_FAILED: EventCategory.EVALUATION,
    RingRiftEventType.ELO_UPDATED: EventCategory.EVALUATION,
    RingRiftEventType.ELO_SIGNIFICANT_CHANGE: EventCategory.EVALUATION,
    RingRiftEventType.ELO_VELOCITY_CHANGED: EventCategory.EVALUATION,
    RingRiftEventType.ADAPTIVE_PARAMS_CHANGED: EventCategory.EVALUATION,

    # Promotion events
    RingRiftEventType.PROMOTION_CANDIDATE: EventCategory.PROMOTION,
    RingRiftEventType.PROMOTION_STARTED: EventCategory.PROMOTION,
    RingRiftEventType.MODEL_PROMOTED: EventCategory.PROMOTION,
    RingRiftEventType.PROMOTION_FAILED: EventCategory.PROMOTION,
    RingRiftEventType.PROMOTION_REJECTED: EventCategory.PROMOTION,
    RingRiftEventType.PROMOTION_ROLLED_BACK: EventCategory.PROMOTION,
    RingRiftEventType.TIER_PROMOTION: EventCategory.PROMOTION,
    RingRiftEventType.CROSSBOARD_PROMOTION: EventCategory.PROMOTION,
    RingRiftEventType.MODEL_UPDATED: EventCategory.PROMOTION,

    # Curriculum events
    RingRiftEventType.CURRICULUM_REBALANCED: EventCategory.CURRICULUM,
    RingRiftEventType.CURRICULUM_ADVANCED: EventCategory.CURRICULUM,
    RingRiftEventType.WEIGHT_UPDATED: EventCategory.CURRICULUM,
    RingRiftEventType.OPPONENT_MASTERED: EventCategory.CURRICULUM,

    # Selfplay events
    RingRiftEventType.SELFPLAY_COMPLETE: EventCategory.SELFPLAY,
    RingRiftEventType.SELFPLAY_TARGET_UPDATED: EventCategory.SELFPLAY,
    RingRiftEventType.SELFPLAY_RATE_CHANGED: EventCategory.SELFPLAY,
    RingRiftEventType.IDLE_RESOURCE_DETECTED: EventCategory.SELFPLAY,

    # Optimization events
    RingRiftEventType.CMAES_TRIGGERED: EventCategory.OPTIMIZATION,
    RingRiftEventType.CMAES_COMPLETED: EventCategory.OPTIMIZATION,
    RingRiftEventType.NAS_TRIGGERED: EventCategory.OPTIMIZATION,
    RingRiftEventType.NAS_STARTED: EventCategory.OPTIMIZATION,
    RingRiftEventType.NAS_GENERATION_COMPLETE: EventCategory.OPTIMIZATION,
    RingRiftEventType.NAS_COMPLETED: EventCategory.OPTIMIZATION,
    RingRiftEventType.NAS_BEST_ARCHITECTURE: EventCategory.OPTIMIZATION,
    RingRiftEventType.PBT_STARTED: EventCategory.OPTIMIZATION,
    RingRiftEventType.PBT_GENERATION_COMPLETE: EventCategory.OPTIMIZATION,
    RingRiftEventType.PBT_COMPLETED: EventCategory.OPTIMIZATION,
    RingRiftEventType.PLATEAU_DETECTED: EventCategory.OPTIMIZATION,
    RingRiftEventType.HYPERPARAMETER_UPDATED: EventCategory.OPTIMIZATION,
    RingRiftEventType.PER_BUFFER_REBUILT: EventCategory.OPTIMIZATION,
    RingRiftEventType.PER_PRIORITIES_UPDATED: EventCategory.OPTIMIZATION,

    # Quality events
    RingRiftEventType.DATA_QUALITY_ALERT: EventCategory.QUALITY,
    RingRiftEventType.QUALITY_CHECK_REQUESTED: EventCategory.QUALITY,
    RingRiftEventType.QUALITY_CHECK_FAILED: EventCategory.QUALITY,
    RingRiftEventType.QUALITY_SCORE_UPDATED: EventCategory.QUALITY,
    RingRiftEventType.QUALITY_DISTRIBUTION_CHANGED: EventCategory.QUALITY,
    RingRiftEventType.HIGH_QUALITY_DATA_AVAILABLE: EventCategory.QUALITY,
    RingRiftEventType.QUALITY_DEGRADED: EventCategory.QUALITY,
    RingRiftEventType.LOW_QUALITY_DATA_WARNING: EventCategory.QUALITY,
    RingRiftEventType.TRAINING_BLOCKED_BY_QUALITY: EventCategory.QUALITY,
    RingRiftEventType.QUALITY_PENALTY_APPLIED: EventCategory.QUALITY,
    RingRiftEventType.QUALITY_FEEDBACK_ADJUSTED: EventCategory.QUALITY,
    RingRiftEventType.SCHEDULER_REGISTERED: EventCategory.QUALITY,
    RingRiftEventType.EXPLORATION_BOOST: EventCategory.QUALITY,
    RingRiftEventType.EXPLORATION_ADJUSTED: EventCategory.QUALITY,

    # Regression events
    RingRiftEventType.REGRESSION_DETECTED: EventCategory.REGRESSION,
    RingRiftEventType.REGRESSION_MINOR: EventCategory.REGRESSION,
    RingRiftEventType.REGRESSION_MODERATE: EventCategory.REGRESSION,
    RingRiftEventType.REGRESSION_SEVERE: EventCategory.REGRESSION,
    RingRiftEventType.REGRESSION_CRITICAL: EventCategory.REGRESSION,
    RingRiftEventType.REGRESSION_CLEARED: EventCategory.REGRESSION,

    # Cluster events
    RingRiftEventType.P2P_MODEL_SYNCED: EventCategory.CLUSTER,
    RingRiftEventType.MODEL_SYNC_REQUESTED: EventCategory.CLUSTER,
    RingRiftEventType.P2P_CLUSTER_HEALTHY: EventCategory.CLUSTER,
    RingRiftEventType.P2P_CLUSTER_UNHEALTHY: EventCategory.CLUSTER,
    RingRiftEventType.P2P_NODES_DEAD: EventCategory.CLUSTER,
    RingRiftEventType.P2P_SELFPLAY_SCALED: EventCategory.CLUSTER,
    RingRiftEventType.CLUSTER_STATUS_CHANGED: EventCategory.CLUSTER,
    RingRiftEventType.CLUSTER_CAPACITY_CHANGED: EventCategory.CLUSTER,
    RingRiftEventType.NODE_UNHEALTHY: EventCategory.CLUSTER,
    RingRiftEventType.NODE_RECOVERED: EventCategory.CLUSTER,
    RingRiftEventType.NODE_ACTIVATED: EventCategory.CLUSTER,
    RingRiftEventType.NODE_CAPACITY_UPDATED: EventCategory.CLUSTER,
    RingRiftEventType.NODE_OVERLOADED: EventCategory.CLUSTER,

    # System events
    RingRiftEventType.DAEMON_STARTED: EventCategory.SYSTEM,
    RingRiftEventType.DAEMON_STOPPED: EventCategory.SYSTEM,
    RingRiftEventType.DAEMON_STATUS_CHANGED: EventCategory.SYSTEM,
    RingRiftEventType.HOST_ONLINE: EventCategory.SYSTEM,
    RingRiftEventType.HOST_OFFLINE: EventCategory.SYSTEM,
    RingRiftEventType.ERROR: EventCategory.SYSTEM,
    RingRiftEventType.HEALTH_CHECK_PASSED: EventCategory.SYSTEM,
    RingRiftEventType.HEALTH_CHECK_FAILED: EventCategory.SYSTEM,
    RingRiftEventType.HEALTH_ALERT: EventCategory.SYSTEM,
    RingRiftEventType.RESOURCE_CONSTRAINT: EventCategory.SYSTEM,
    RingRiftEventType.RESOURCE_CONSTRAINT_DETECTED: EventCategory.SYSTEM,
    RingRiftEventType.RECOVERY_INITIATED: EventCategory.SYSTEM,
    RingRiftEventType.RECOVERY_COMPLETED: EventCategory.SYSTEM,
    RingRiftEventType.RECOVERY_FAILED: EventCategory.SYSTEM,
    RingRiftEventType.MODEL_CORRUPTED: EventCategory.SYSTEM,
    RingRiftEventType.COORDINATOR_HEALTHY: EventCategory.SYSTEM,
    RingRiftEventType.COORDINATOR_UNHEALTHY: EventCategory.SYSTEM,
    RingRiftEventType.COORDINATOR_HEALTH_DEGRADED: EventCategory.SYSTEM,
    RingRiftEventType.COORDINATOR_SHUTDOWN: EventCategory.SYSTEM,
    RingRiftEventType.COORDINATOR_INIT_FAILED: EventCategory.SYSTEM,
    RingRiftEventType.COORDINATOR_HEARTBEAT: EventCategory.SYSTEM,
    RingRiftEventType.HANDLER_TIMEOUT: EventCategory.SYSTEM,
    RingRiftEventType.HANDLER_FAILED: EventCategory.SYSTEM,
    RingRiftEventType.REGISTRY_UPDATED: EventCategory.SYSTEM,
    RingRiftEventType.METRICS_UPDATED: EventCategory.SYSTEM,
    RingRiftEventType.CACHE_INVALIDATED: EventCategory.SYSTEM,
    RingRiftEventType.PARITY_VALIDATION_STARTED: EventCategory.SYSTEM,
    RingRiftEventType.PARITY_VALIDATION_COMPLETED: EventCategory.SYSTEM,
    RingRiftEventType.PARITY_FAILURE_RATE_CHANGED: EventCategory.SYSTEM,
    RingRiftEventType.ENCODING_BATCH_COMPLETED: EventCategory.SYSTEM,
    RingRiftEventType.CALIBRATION_COMPLETED: EventCategory.SYSTEM,

    # Work queue events
    RingRiftEventType.WORK_QUEUED: EventCategory.WORK,
    RingRiftEventType.WORK_CLAIMED: EventCategory.WORK,
    RingRiftEventType.WORK_STARTED: EventCategory.WORK,
    RingRiftEventType.WORK_COMPLETED: EventCategory.WORK,
    RingRiftEventType.WORK_FAILED: EventCategory.WORK,
    RingRiftEventType.WORK_RETRY: EventCategory.WORK,
    RingRiftEventType.WORK_TIMEOUT: EventCategory.WORK,
    RingRiftEventType.WORK_CANCELLED: EventCategory.WORK,
    RingRiftEventType.JOB_PREEMPTED: EventCategory.WORK,

    # Lock/sync events
    RingRiftEventType.LOCK_ACQUIRED: EventCategory.SYNC,
    RingRiftEventType.LOCK_RELEASED: EventCategory.SYNC,
    RingRiftEventType.LOCK_TIMEOUT: EventCategory.SYNC,
    RingRiftEventType.DEADLOCK_DETECTED: EventCategory.SYNC,
    RingRiftEventType.CHECKPOINT_SAVED: EventCategory.SYNC,
    RingRiftEventType.CHECKPOINT_LOADED: EventCategory.SYNC,

    # Task lifecycle events
    RingRiftEventType.TASK_SPAWNED: EventCategory.TASK,
    RingRiftEventType.TASK_HEARTBEAT: EventCategory.TASK,
    RingRiftEventType.TASK_COMPLETED: EventCategory.TASK,
    RingRiftEventType.TASK_FAILED: EventCategory.TASK,
    RingRiftEventType.TASK_ORPHANED: EventCategory.TASK,
    RingRiftEventType.TASK_CANCELLED: EventCategory.TASK,
    RingRiftEventType.TASK_ABANDONED: EventCategory.TASK,
    RingRiftEventType.BACKPRESSURE_ACTIVATED: EventCategory.TASK,
    RingRiftEventType.BACKPRESSURE_RELEASED: EventCategory.TASK,

    # Leader election events
    RingRiftEventType.LEADER_ELECTED: EventCategory.CLUSTER,
    RingRiftEventType.LEADER_LOST: EventCategory.CLUSTER,
    RingRiftEventType.LEADER_STEPDOWN: EventCategory.CLUSTER,

    # Stage events
    RingRiftEventType.STAGE_SELFPLAY_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_CANONICAL_SELFPLAY_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_GPU_SELFPLAY_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_SYNC_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_PARITY_VALIDATION_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_NPZ_EXPORT_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_TRAINING_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_TRAINING_STARTED: EventCategory.STAGE,
    RingRiftEventType.STAGE_TRAINING_FAILED: EventCategory.STAGE,
    RingRiftEventType.STAGE_EVALUATION_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_SHADOW_TOURNAMENT_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_ELO_CALIBRATION_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_CMAES_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_PBT_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_NAS_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_PROMOTION_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_TIER_GATING_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_ITERATION_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_CLUSTER_SYNC_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_MODEL_SYNC_COMPLETE: EventCategory.STAGE,
}


# =============================================================================
# Backwards Compatibility Aliases
# =============================================================================

# Alias the old names to the new unified type
DataEventType = RingRiftEventType


# Create a StageEvent alias class that maps to RingRiftEventType
class StageEvent(Enum):
    """Backwards compatibility alias for stage events.

    .. deprecated:: December 2025
        Use RingRiftEventType.STAGE_* instead.
    """

    SELFPLAY_COMPLETE = RingRiftEventType.STAGE_SELFPLAY_COMPLETE.value
    CANONICAL_SELFPLAY_COMPLETE = RingRiftEventType.STAGE_CANONICAL_SELFPLAY_COMPLETE.value
    GPU_SELFPLAY_COMPLETE = RingRiftEventType.STAGE_GPU_SELFPLAY_COMPLETE.value
    SYNC_COMPLETE = RingRiftEventType.STAGE_SYNC_COMPLETE.value
    PARITY_VALIDATION_COMPLETE = RingRiftEventType.STAGE_PARITY_VALIDATION_COMPLETE.value
    NPZ_EXPORT_COMPLETE = RingRiftEventType.STAGE_NPZ_EXPORT_COMPLETE.value
    TRAINING_COMPLETE = RingRiftEventType.STAGE_TRAINING_COMPLETE.value
    TRAINING_STARTED = RingRiftEventType.STAGE_TRAINING_STARTED.value
    TRAINING_FAILED = RingRiftEventType.STAGE_TRAINING_FAILED.value
    EVALUATION_COMPLETE = RingRiftEventType.STAGE_EVALUATION_COMPLETE.value
    SHADOW_TOURNAMENT_COMPLETE = RingRiftEventType.STAGE_SHADOW_TOURNAMENT_COMPLETE.value
    ELO_CALIBRATION_COMPLETE = RingRiftEventType.STAGE_ELO_CALIBRATION_COMPLETE.value
    CMAES_COMPLETE = RingRiftEventType.STAGE_CMAES_COMPLETE.value
    PBT_COMPLETE = RingRiftEventType.STAGE_PBT_COMPLETE.value
    NAS_COMPLETE = RingRiftEventType.STAGE_NAS_COMPLETE.value
    PROMOTION_COMPLETE = RingRiftEventType.STAGE_PROMOTION_COMPLETE.value
    TIER_GATING_COMPLETE = RingRiftEventType.STAGE_TIER_GATING_COMPLETE.value
    ITERATION_COMPLETE = RingRiftEventType.STAGE_ITERATION_COMPLETE.value
    CLUSTER_SYNC_COMPLETE = RingRiftEventType.STAGE_CLUSTER_SYNC_COMPLETE.value
    MODEL_SYNC_COMPLETE = RingRiftEventType.STAGE_MODEL_SYNC_COMPLETE.value


# =============================================================================
# Utility Functions
# =============================================================================

def get_events_by_category(category: EventCategory) -> list[RingRiftEventType]:
    """Get all event types in a category."""
    return [
        event_type
        for event_type, cat in _EVENT_CATEGORIES.items()
        if cat == category
    ]


def is_cross_process_event(event_type: RingRiftEventType) -> bool:
    """Check if an event should be propagated across processes.

    These events are important for distributed coordination.
    """
    return event_type in CROSS_PROCESS_EVENT_TYPES


# Events that should be propagated across processes
CROSS_PROCESS_EVENT_TYPES = {
    # Success events - coordination across processes
    RingRiftEventType.MODEL_PROMOTED,
    RingRiftEventType.TIER_PROMOTION,
    RingRiftEventType.TRAINING_STARTED,
    RingRiftEventType.TRAINING_COMPLETED,
    RingRiftEventType.EVALUATION_COMPLETED,
    RingRiftEventType.CURRICULUM_REBALANCED,
    RingRiftEventType.CURRICULUM_ADVANCED,
    RingRiftEventType.SELFPLAY_TARGET_UPDATED,
    RingRiftEventType.ELO_SIGNIFICANT_CHANGE,
    RingRiftEventType.P2P_MODEL_SYNCED,
    RingRiftEventType.PLATEAU_DETECTED,
    RingRiftEventType.DATA_SYNC_COMPLETED,
    RingRiftEventType.HYPERPARAMETER_UPDATED,
    RingRiftEventType.GAME_SYNCED,
    RingRiftEventType.DATA_STALE,

    # Failure events
    RingRiftEventType.TRAINING_FAILED,
    RingRiftEventType.EVALUATION_FAILED,
    RingRiftEventType.PROMOTION_FAILED,
    RingRiftEventType.DATA_SYNC_FAILED,

    # Host/cluster events
    RingRiftEventType.HOST_ONLINE,
    RingRiftEventType.HOST_OFFLINE,
    RingRiftEventType.DAEMON_STARTED,
    RingRiftEventType.DAEMON_STOPPED,
    RingRiftEventType.DAEMON_STATUS_CHANGED,

    # Cluster health events (December 2025 - Phase 21)
    RingRiftEventType.NODE_UNHEALTHY,
    RingRiftEventType.NODE_RECOVERED,
    RingRiftEventType.P2P_CLUSTER_HEALTHY,
    RingRiftEventType.P2P_CLUSTER_UNHEALTHY,
    RingRiftEventType.HEALTH_CHECK_PASSED,
    RingRiftEventType.HEALTH_CHECK_FAILED,
    RingRiftEventType.HEALTH_ALERT,

    # Trigger events
    RingRiftEventType.CMAES_TRIGGERED,
    RingRiftEventType.NAS_TRIGGERED,
    RingRiftEventType.TRAINING_THRESHOLD_REACHED,
    RingRiftEventType.CACHE_INVALIDATED,

    # Regression events
    RingRiftEventType.REGRESSION_DETECTED,
    RingRiftEventType.REGRESSION_SEVERE,
    RingRiftEventType.REGRESSION_CRITICAL,
    RingRiftEventType.REGRESSION_CLEARED,
}


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main enum
    "RingRiftEventType",
    # Category support
    "EventCategory",
    "get_events_by_category",
    # Cross-process support
    "CROSS_PROCESS_EVENT_TYPES",
    "is_cross_process_event",
    # Backwards compatibility
    "DataEventType",
    "StageEvent",
]
