"""Data Pipeline Event System.

.. deprecated:: December 2025
    This module is being superseded by the unified event router.
    For new code, prefer using:

        from app.coordination.event_router import (
            get_router, publish, subscribe, DataEventType
        )

    The unified router provides:
    - Automatic routing to all event buses (data, stage, cross-process)
    - Cross-process event persistence
    - Event flow auditing
    - Unified subscription management

    This module remains fully functional for backwards compatibility.

This module provides an event bus for coordinating components of the
AI self-improvement loop. Events allow loose coupling between:
- Data collection
- Training triggers
- Evaluation
- Model promotion
- Curriculum rebalancing

Usage:
    from app.coordination.event_router import DataEventType, DataEvent, get_event_bus

    # Subscribe to events
    bus = get_event_bus()
    bus.subscribe(DataEventType.NEW_GAMES_AVAILABLE, handle_new_games)

    # Publish events
    await bus.publish(DataEvent(
        event_type=DataEventType.NEW_GAMES_AVAILABLE,
        payload={"host": "gh200-a", "new_games": 500}
    ))
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

__all__ = [
    "DataEvent",
    # Event types
    "DataEventType",
    # Event bus
    "EventBus",
    "get_event_bus",
    "reset_event_bus",
    # Generic emitter (used by consolidated event_router)
    "emit_data_event",
    # Cluster health emitters (December 2025 - Phase 21)
    "emit_node_unhealthy",
    "emit_node_recovered",
    "emit_health_check_passed",
    "emit_health_check_failed",
    "emit_p2p_cluster_healthy",
    "emit_p2p_cluster_unhealthy",
    "emit_p2p_restarted",  # Dec 30, 2025: P2P restart completed
    # Task lifecycle emitters (December 2025 Wave 2)
    "emit_task_abandoned",
    # Job spawn verification emitters (January 2026 - Sprint 6)
    "emit_job_spawn_verified",
    "emit_job_spawn_failed",
    # Disk space management emitters (December 2025)
    "emit_disk_space_low",
    # Leader heartbeat monitoring (P0 Dec 2025)
    "emit_leader_heartbeat_missing",
    # Leader lease expiry (Jan 2026)
    "emit_leader_lease_expired",
]

# Global singleton instance
_event_bus: EventBus | None = None


class DataEventType(Enum):
    """Types of data pipeline events."""

    # Data collection events
    NEW_GAMES_AVAILABLE = "new_games"
    DATA_SYNC_STARTED = "sync_started"
    DATA_SYNC_COMPLETED = "sync_completed"
    DATA_SYNC_FAILED = "sync_failed"
    QUEUED_EVENTS_SYNCED = "queued_events_synced"  # Jan 4, 2026: Fallback queue events synced to coordinator
    GAME_SYNCED = "game_synced"  # Individual game(s) synced to targets

    # Data freshness events
    DATA_STALE = "data_stale"  # Training data is stale
    DATA_FRESH = "data_fresh"  # Training data is fresh
    DATA_STARVATION_CRITICAL = "data_starvation_critical"  # Jan 5, 2026: Config has critically low game count (<20), needs priority dispatch
    IDLE_NODE_WORK_INJECTED = "idle_node_work_injected"  # Jan 5, 2026: Work injected for idle GPU nodes
    SYNC_TRIGGERED = "sync_triggered"  # Sync triggered due to stale data
    SYNC_REQUEST = "sync_request"  # Explicit sync request (router-driven)

    # Data consolidation events (December 2025 - fix training pipeline)
    CONSOLIDATION_STARTED = "consolidation_started"  # Consolidation in progress
    CONSOLIDATION_COMPLETE = "consolidation_complete"  # Games merged to canonical DB

    # NPZ export events (December 2025 - triggers combination daemon)
    NPZ_EXPORT_STARTED = "npz_export_started"  # Export in progress
    NPZ_EXPORT_COMPLETE = "npz_export_complete"  # NPZ export finished
    # Sprint 4 (Jan 2, 2026): Export validation pre-check
    EXPORT_VALIDATION_FAILED = "export_validation_failed"  # Export blocked by validation

    # NPZ combination events (December 2025 - quality-weighted data combination)
    NPZ_COMBINATION_STARTED = "npz_combination_started"  # Combination in progress
    NPZ_COMBINATION_COMPLETE = "npz_combination_complete"  # Combined NPZ ready
    NPZ_COMBINATION_FAILED = "npz_combination_failed"  # Combination failed

    # Training events
    TRAINING_THRESHOLD_REACHED = "training_threshold"
    TRAINING_STARTED = "training_started"
    TRAINING_PROGRESS = "training_progress"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_FAILED = "training_failed"
    TRAINING_LOCK_ACQUIRED = "training_lock_acquired"  # Dec 30, 2025: Training lock acquired for config
    TRAINING_LOCK_TIMEOUT = "training_lock_timeout"  # Jan 4, 2026: Training lock auto-released due to TTL expiry
    TRAINING_PROCESS_KILLED = "training_process_killed"  # Jan 4, 2026: Stuck training process killed by watchdog
    TRAINING_HEARTBEAT = "training_heartbeat"  # Jan 4, 2026: Training process heartbeat for watchdog monitoring
    TRAINING_SLOT_UNAVAILABLE = "training_slot_unavailable"  # Dec 30, 2025: Training slot not available
    TRAINING_TIMEOUT_REACHED = "training_timeout_reached"  # Jan 3, 2026: Training exceeded time limit
    TRAINING_DATA_RECOVERED = "training_data_recovered"  # Jan 3, 2026 Sprint 13.3: NPZ re-exported after corruption
    TRAINING_DATA_RECOVERY_FAILED = "training_data_recovery_failed"  # Jan 3, 2026 Sprint 13.3: NPZ re-export failed
    TRAINING_REQUESTED = "training_requested"  # Jan 5, 2026: Request training for a config (from cascade/scheduler)

    # Cascade training events (Jan 5, 2026 - Task 8.7: Cross-architecture curriculum)
    # Automates transfer learning: 2p → 3p → 4p when Elo thresholds are met
    CASCADE_TRANSFER_TRIGGERED = "cascade_transfer_triggered"  # Transfer learning triggered between player counts
    CASCADE_TRANSFER_FAILED = "cascade_transfer_failed"  # Transfer learning failed

    # Evaluation events
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_PROGRESS = "evaluation_progress"
    EVALUATION_COMPLETED = "evaluation_completed"
    EVALUATION_FAILED = "evaluation_failed"
    EVALUATION_BACKPRESSURE = "evaluation_backpressure"  # Dec 29, 2025: Eval queue backlogged, pause training
    EVALUATION_BACKPRESSURE_RELEASED = "evaluation_backpressure_released"  # Dec 29, 2025: Eval queue drained, resume training
    MODEL_EVALUATION_BLOCKED = "model_evaluation_blocked"  # Dec 2025 Phase 3: Model not distributed for eval
    ELO_UPDATED = "elo_updated"
    HARNESS_EVALUATION_COMPLETED = "harness_evaluation_completed"  # Dec 31, 2025: Per-harness Elo tracking
    # Sprint 13 Session 4 (Jan 3, 2026): Model evaluation automation
    EVALUATION_REQUESTED = "evaluation_requested"  # Request to evaluate a model (from scanner daemon)
    EVALUATION_QUEUED = "evaluation_queued"  # Model added to persistent eval queue
    EVALUATION_RECOVERED = "evaluation_recovered"  # Stuck evaluation recovered and requeued
    EVALUATION_STUCK = "evaluation_stuck"  # Evaluation stuck detected (exceeded timeout)
    EVALUATION_SUBMITTED = "evaluation_submitted"  # Jan 3, 2026: Eval result submitted to hashgraph consensus
    UNEVALUATED_MODELS_FOUND = "unevaluated_models_found"  # Scanner found models without Elo ratings
    # January 6, 2026: Head-to-head evaluation against previous model version
    HEAD_TO_HEAD_COMPLETED = "head_to_head_completed"  # New model evaluated vs previous best for same config
    # January 9, 2026: Periodic multi-harness and cross-config tournament events
    MULTI_HARNESS_EVALUATION_COMPLETED = "multi_harness_evaluation_completed"  # All harnesses evaluated for model
    CROSS_CONFIG_TOURNAMENT_COMPLETED = "cross_config_tournament_completed"  # Cross-config family tournament done
    TOPN_ROUNDROBIN_COMPLETED = "topn_roundrobin_completed"  # Top-N round-robin tournament completed (Jan 2026)

    # Promotion events
    PROMOTION_CANDIDATE = "promotion_candidate"
    PROMOTION_STARTED = "promotion_started"
    MODEL_PROMOTED = "model_promoted"
    PROMOTION_FAILED = "promotion_failed"
    PROMOTION_REJECTED = "promotion_rejected"
    MODEL_UPDATED = "model_updated"  # Model metadata or path updated (pre-promotion)
    PROMOTION_CONSENSUS_APPROVED = "promotion_consensus_approved"  # Jan 3, 2026: BFT promotion via hashgraph

    # Curriculum events
    CURRICULUM_REBALANCED = "curriculum_rebalanced"
    CURRICULUM_ADVANCED = "curriculum_advanced"  # Move to harder curriculum tier
    CURRICULUM_ADVANCEMENT_NEEDED = "curriculum_advancement_needed"  # Dec 29, 2025: Signal to advance curriculum (from stagnant Elo)
    CURRICULUM_PROPAGATE = "curriculum_propagate"  # Jan 2026: Propagate curriculum advancement to similar configs
    CURRICULUM_ROLLBACK = "curriculum_rollback"  # Session 17.25: Signal to restore prior curriculum weights after regression
    CURRICULUM_ROLLBACK_COMPLETED = "curriculum_rollback_completed"  # Sprint 16.1: Confirm curriculum weight rollback after regression
    WEIGHT_UPDATED = "weight_updated"
    ELO_SIGNIFICANT_CHANGE = "elo_significant_change"  # Triggers curriculum rebalance

    # Selfplay feedback events
    SELFPLAY_COMPLETE = "selfplay_complete"  # P0.2 Dec 2025: Selfplay batch finished
    SELFPLAY_TARGET_UPDATED = "selfplay_target_updated"  # Request more/fewer games
    SELFPLAY_RATE_CHANGED = "selfplay_rate_changed"  # Phase 19.3: Rate multiplier changed (>20%)
    SELFPLAY_ALLOCATION_UPDATED = "selfplay_allocation_updated"  # Dec 2025: Allocation changed (exploration boost, curriculum)

    # Batch scheduling events (Dec 27, 2025 - P0 pipeline coordination fix)
    BATCH_SCHEDULED = "batch_scheduled"  # Batch of jobs selected for dispatch
    BATCH_DISPATCHED = "batch_dispatched"  # Batch of jobs sent to nodes

    # Optimization events
    CMAES_TRIGGERED = "cmaes_triggered"
    CMAES_COMPLETED = "cmaes_completed"
    NAS_TRIGGERED = "nas_triggered"
    PLATEAU_DETECTED = "plateau_detected"
    HYPERPARAMETER_UPDATED = "hyperparameter_updated"

    # Elo momentum events (December 2025 - Phase 15)
    ELO_VELOCITY_CHANGED = "elo_velocity_changed"  # Significant change in Elo improvement rate
    ADAPTIVE_PARAMS_CHANGED = "adaptive_params_changed"  # Training params adjusted based on Elo

    # PBT events
    PBT_STARTED = "pbt_started"
    PBT_GENERATION_COMPLETE = "pbt_generation_complete"
    PBT_COMPLETED = "pbt_completed"

    # NAS events
    NAS_STARTED = "nas_started"
    NAS_GENERATION_COMPLETE = "nas_generation_complete"
    NAS_COMPLETED = "nas_completed"
    NAS_BEST_ARCHITECTURE = "nas_best_architecture"

    # Architecture feedback events (December 29, 2025)
    ARCHITECTURE_WEIGHTS_UPDATED = "architecture_weights_updated"  # Allocation weights recalculated

    # PER (Prioritized Experience Replay) events
    PER_BUFFER_REBUILT = "per_buffer_rebuilt"
    PER_PRIORITIES_UPDATED = "per_priorities_updated"

    # Tier gating events
    TIER_PROMOTION = "tier_promotion"
    CROSSBOARD_PROMOTION = "crossboard_promotion"  # Multi-config promotion decision

    # Parity validation events
    PARITY_VALIDATION_STARTED = "parity_validation_started"
    PARITY_VALIDATION_COMPLETED = "parity_validation_completed"

    # Data quality events
    DATA_QUALITY_ALERT = "data_quality_alert"
    QUALITY_CHECK_REQUESTED = "quality_check_requested"  # Phase 9: Request on-demand quality check
    QUALITY_CHECK_FAILED = "quality_check_failed"
    QUALITY_SCORE_UPDATED = "quality_score_updated"  # Game quality recalculated
    QUALITY_DISTRIBUTION_CHANGED = "quality_distribution_changed"  # Significant shift
    HIGH_QUALITY_DATA_AVAILABLE = "high_quality_data_available"  # Ready for training
    QUALITY_DEGRADED = "quality_degraded"  # Phase 5: Quality dropped below threshold
    LOW_QUALITY_DATA_WARNING = "low_quality_data_warning"  # Below threshold
    TRAINING_BLOCKED_BY_QUALITY = "training_blocked_by_quality"  # Quality too low to train
    # P0.6 Dec 2025: Added missing event types to enable type-safe subscriptions
    QUALITY_FEEDBACK_ADJUSTED = "quality_feedback_adjusted"  # Quality feedback updated for config
    SCHEDULER_REGISTERED = "scheduler_registered"  # Temperature scheduler registered for config
    QUALITY_PENALTY_APPLIED = "quality_penalty_applied"  # Penalty applied → reduce selfplay rate
    # Jan 5, 2026 - Session 17.29: Quality-triggered auto-dispatch
    QUALITY_SELFPLAY_DISPATCHED = "quality_selfplay_dispatched"  # High-quality selfplay auto-dispatched

    # P2P recovery events
    VOTER_HEALING_REQUESTED = "voter_healing_requested"  # Jan 2026: Voter healing requested by P2PRecoveryDaemon
    # P1.4 Dec 2025: Added orphaned event types to enable type-safe subscriptions
    EXPLORATION_BOOST = "exploration_boost"  # Request to boost exploration temperature
    EXPLORATION_ADJUSTED = "exploration_adjusted"  # Exploration strategy changed (from FeedbackSignals)
    OPPONENT_MASTERED = "opponent_mastered"  # Opponent mastered → advance curriculum

    # Training loss monitoring events (December 2025)
    TRAINING_LOSS_ANOMALY = "training_loss_anomaly"  # Unusual loss spike/drop detected
    TRAINING_LOSS_TREND = "training_loss_trend"  # Loss trend changed (improving/degrading)

    # Registry & metrics events
    REGISTRY_UPDATED = "registry_updated"
    METRICS_UPDATED = "metrics_updated"
    CACHE_INVALIDATED = "cache_invalidated"

    # Regression detection events (from unified RegressionDetector)
    REGRESSION_DETECTED = "regression_detected"  # Any regression
    REGRESSION_MINOR = "regression_minor"  # Severity: minor
    REGRESSION_MODERATE = "regression_moderate"  # Severity: moderate
    REGRESSION_SEVERE = "regression_severe"  # Severity: severe
    REGRESSION_CRITICAL = "regression_critical"  # Severity: critical - rollback recommended
    REGRESSION_CLEARED = "regression_cleared"  # Model recovered from regression

    # Model import events (Sprint 13 Session 4 - Jan 3, 2026)
    MODEL_IMPORTED = "model_imported"  # Model imported from OWC drive
    OWC_MODELS_DISCOVERED = "owc_models_discovered"  # OWC scan discovered models (observability)

    # Backlog evaluation events (Sprint 15 - Jan 3, 2026)
    OWC_MODEL_DISCOVERED = "owc_model_discovered"  # Single model found on OWC drive
    OWC_MODEL_BACKLOG_QUEUED = "owc_model_backlog_queued"  # Model queued for evaluation
    OWC_MODEL_EVALUATION_STARTED = "owc_model_evaluation_started"  # Backlog evaluation began
    OWC_MODEL_EVALUATION_COMPLETED = "owc_model_evaluation_completed"  # Backlog evaluation finished
    OWC_MODEL_EVALUATION_FAILED = "owc_model_evaluation_failed"  # Backlog evaluation failed
    BACKLOG_DISCOVERY_COMPLETED = "backlog_discovery_completed"  # Discovery cycle complete

    # P2P/Model sync events
    P2P_MODEL_SYNCED = "p2p_model_synced"
    MODEL_SYNC_REQUESTED = "model_sync_requested"
    MODEL_DISTRIBUTION_STARTED = "model_distribution_started"  # Dec 2025: Model distribution initiated
    MODEL_DISTRIBUTION_COMPLETE = "model_distribution_complete"  # Dec 2025: Model distributed to cluster
    MODEL_DISTRIBUTION_FAILED = "model_distribution_failed"  # Dec 2025: Model distribution failed
    MODEL_NOT_FOUND = "model_not_found"  # Jan 2026: Model file missing at dispatch time
    DISTRIBUTION_INCOMPLETE = "distribution_incomplete"  # Dec 2025 Phase 3: Model not on enough nodes
    P2P_CLUSTER_HEALTHY = "p2p_cluster_healthy"
    P2P_CLUSTER_UNHEALTHY = "p2p_cluster_unhealthy"
    CLUSTER_HEALTH_CHANGED = "cluster_health_changed"  # January 2026: Unified cluster health state change
    SYNC_STALLED = "sync_stalled"  # December 2025: Sync operation stalled/timed out
    SYNC_CHECKSUM_FAILED = "sync_checksum_failed"  # December 2025: Checksum mismatch after sync
    SYNC_NODE_UNREACHABLE = "sync_node_unreachable"  # December 2025: Node unreachable for sync
    SYNC_FAILURE_CRITICAL = "sync_failure_critical"  # December 2025: Multiple consecutive sync failures
    P2P_NODE_DEAD = "p2p_node_dead"  # Dec 2025: Single node dead (vs P2P_NODES_DEAD for batch)
    P2P_NODES_DEAD = "p2p_nodes_dead"
    P2P_SELFPLAY_SCALED = "p2p_selfplay_scaled"
    P2P_RESTART_TRIGGERED = "p2p_restart_triggered"  # Dec 2025: P2P orchestrator restart initiated
    P2P_RESTARTED = "p2p_restarted"  # Dec 30, 2025: P2P orchestrator successfully restarted
    P2P_HEALTH_RECOVERED = "p2p_health_recovered"  # Dec 2025: P2P cluster recovered from unhealthy
    NETWORK_ISOLATION_DETECTED = "network_isolation_detected"  # Dec 2025: P2P sees fewer peers than Tailscale
    QUORUM_PRIORITY_RECONNECT = "quorum_priority_reconnect"  # Dec 30, 2025: Priority reconnection order for quorum
    CLUSTER_P2P_RECOVERY_COMPLETED = "cluster_p2p_recovery_completed"  # Dec 31, 2025: Cluster-wide SSH recovery
    REMOTE_P2P_RECOVERY_SUCCESS = "remote_p2p_recovery_success"  # Session 17.25: Node successfully recovered via SSH
    REMOTE_P2P_RECOVERY_FAILED = "remote_p2p_recovery_failed"  # Session 17.25: Recovery attempt failed

    # Voter health events (December 30, 2025 - 48h autonomous operation)
    VOTER_OFFLINE = "voter_offline"  # Individual voter became unreachable
    VOTER_ONLINE = "voter_online"  # Individual voter recovered
    VOTER_DEMOTED = "voter_demoted"  # Jan 2026: Voter demoted due to health issues (audit trail)
    VOTER_PROMOTED = "voter_promoted"  # Jan 2026: Voter re-promoted after recovery (audit trail)
    VOTER_FLAPPING = "voter_flapping"  # Jan 2026 Sprint 15: Voter is unstable (frequent online/offline)
    QUORUM_LOST = "quorum_lost"  # Quorum threshold crossed (was OK, now lost)
    QUORUM_RESTORED = "quorum_restored"  # Quorum threshold crossed (was lost, now OK)
    QUORUM_AT_RISK = "quorum_at_risk"  # Quorum marginal (e.g., exactly at threshold)
    QUORUM_RECOVERY_STARTED = "quorum_recovery_started"  # Recovery initiated after quorum lost
    QUORUM_VALIDATION_FAILED = "quorum_validation_failed"  # Jan 4, 2026: Pre-startup quorum check failed

    # Partition healing events (January 2026)
    PARTITION_HEALING_STARTED = "partition_healing_started"  # Healing pass initiated
    PARTITION_HEALED = "partition_healed"  # Partitions successfully healed
    PARTITION_HEALING_FAILED = "partition_healing_failed"  # Healing attempt failed
    HEALING_CONVERGENCE_FAILED = "healing_convergence_failed"  # Jan 4, 2026: Healing reported success but gossip didn't converge
    P2P_RECOVERY_NEEDED = "p2p_recovery_needed"  # Jan 3, 2026: Max escalation reached, manual intervention needed
    P2P_ZOMBIE_DETECTED = "p2p_zombie_detected"  # Jan 7, 2026: HTTP server crashed but process continues, terminating
    P2P_LOOP_STARTUP_FAILED = "p2p_loop_startup_failed"  # Jan 7, 2026: P2P background loop failed to start within timeout
    P2P_LOOP_PERFORMANCE_DEGRADED = "p2p_loop_performance_degraded"  # Jan 7, 2026: Loop avg run duration exceeds 50% of interval
    GOSSIP_STATE_CLEANUP_COMPLETED = "gossip_state_cleanup_completed"  # Jan 7, 2026: Gossip state TTL cleanup completed

    # Progress monitoring events (December 2025 - 48h autonomous operation)
    PROGRESS_STALL_DETECTED = "progress_stall_detected"  # Config Elo stalled, recovery triggered
    PROGRESS_RECOVERED = "progress_recovered"  # Config resumed making Elo progress

    # Orphan detection events
    ORPHAN_GAMES_DETECTED = "orphan_games_detected"  # Unregistered game databases found
    ORPHAN_GAMES_REGISTERED = "orphan_games_registered"  # Orphans auto-registered

    # Replication repair events (December 2025)
    REPAIR_COMPLETED = "repair_completed"  # Repair job succeeded
    REPAIR_FAILED = "repair_failed"  # Repair job failed
    REPLICATION_ALERT = "replication_alert"  # Replication health alert

    # Database lifecycle events (Phase 4A.3 - December 2025)
    DATABASE_CREATED = "database_created"  # New database file created - immediate registration

    # System events
    DAEMON_STARTED = "daemon_started"
    DAEMON_STOPPED = "daemon_stopped"
    DAEMON_STATUS_CHANGED = "daemon_status_changed"  # Watchdog alerts (stuck, crashed, restarted)
    DAEMON_PERMANENTLY_FAILED = "daemon_permanently_failed"  # Dec 2025: Exceeded hourly restart limit
    DAEMON_CRASH_LOOP_DETECTED = "daemon_crash_loop_detected"  # Dec 2025: Early warning before permanent failure
    # Jan 2026 Sprint 17.9: Unified failure classification events
    DAEMON_FAILURE_CLASSIFIED = "daemon_failure_classified"  # Failure category determined (transient/degraded/persistent/critical)
    DAEMON_FAILURE_ESCALATED = "daemon_failure_escalated"  # Failure category increased in severity
    DAEMON_FAILURE_RECOVERED = "daemon_failure_recovered"  # Daemon recovered from failure state
    HOST_ONLINE = "host_online"
    HOST_OFFLINE = "host_offline"
    ERROR = "error"

    # Config sync events (December 2025 - distributed config propagation)
    CONFIG_UPDATED = "config_updated"  # Cluster config (distributed_hosts.yaml) changed
    CONFIG_DIVERGENCE_DETECTED = "config_divergence_detected"  # Node has different config than peers

    # Health & Recovery events (Phase 10 consolidation)
    HEALTH_CHECK_PASSED = "health_check_passed"
    HEALTH_CHECK_FAILED = "health_check_failed"
    HEALTH_ALERT = "health_alert"  # General health warning
    RESOURCE_CONSTRAINT = "resource_constraint"  # CPU/GPU/Memory/Disk pressure
    MEMORY_PRESSURE = "memory_pressure"  # Dec 29, 2025: GPU VRAM or process RSS critical - pause spawning
    SOCKET_LEAK_DETECTED = "socket_leak_detected"  # Jan 2026: Socket/FD leak detected
    SOCKET_LEAK_RECOVERED = "socket_leak_recovered"  # Jan 2026: Socket/FD leak recovered
    P2P_CONNECTION_RESET_REQUESTED = "p2p_connection_reset_requested"  # Jan 2026: Request P2P reset
    TAILSCALE_CLI_ERROR = "tailscale_cli_error"  # Jan 2026: Tailscale CLI command failed
    TAILSCALE_CLI_RECOVERED = "tailscale_cli_recovered"  # Jan 2026: Tailscale CLI working again
    GPU_CAPABILITY_MISMATCH = "gpu_capability_mismatch"  # Jan 8, 2026: Configured role expects GPU but none detected
    PYTORCH_CUDA_MISMATCH = "pytorch_cuda_mismatch"  # Jan 9, 2026: GPU detected but PyTorch has no CUDA support
    NODE_OVERLOADED = "node_overloaded"  # Node resource overload (job redistribution)
    RECOVERY_INITIATED = "recovery_initiated"  # Auto-recovery started
    RECOVERY_COMPLETED = "recovery_completed"  # Auto-recovery finished
    RECOVERY_FAILED = "recovery_failed"  # Auto-recovery failed

    # Work Queue events (December 2025 - coordination integration)
    WORK_QUEUED = "work_queued"  # New work added to queue
    WORK_CLAIMED = "work_claimed"  # Work claimed by a node
    WORK_STARTED = "work_started"  # Work execution started
    WORK_COMPLETED = "work_completed"  # Work completed successfully
    WORK_FAILED = "work_failed"  # Work failed permanently
    WORK_RETRY = "work_retry"  # Work failed, will retry
    WORK_TIMEOUT = "work_timeout"  # Work timed out
    WORK_CANCELLED = "work_cancelled"  # Work cancelled
    JOB_PREEMPTED = "job_preempted"  # Job preempted for higher priority work
    # Jan 2026: Work queue stall detection for autonomous operation
    WORK_QUEUE_STALLED = "work_queue_stalled"  # No work dispatched for extended period
    WORK_QUEUE_RECOVERED = "work_queue_recovered"  # Work dispatch resumed after stall
    WORK_QUEUE_EXHAUSTED = "work_queue_exhausted"  # Jan 4, 2026: Work queue completely empty

    # Jan 5, 2026: Work queue backpressure events (Phase 2 - optimization)
    WORK_QUEUE_BACKPRESSURE = "work_queue_backpressure"  # Queue capacity at high level
    WORK_QUEUE_BACKPRESSURE_RELEASED = "work_queue_backpressure_released"  # Queue capacity released

    # Jan 4, 2026: Autonomous queue fallback (Phase 2 - P2P resilience)
    AUTONOMOUS_QUEUE_ACTIVATED = "autonomous_queue_activated"  # Fallback mode enabled
    AUTONOMOUS_QUEUE_DEACTIVATED = "autonomous_queue_deactivated"  # Normal mode restored

    # Jan 4, 2026: Underutilization recovery (Phase 3 - P2P resilience)
    UTILIZATION_RECOVERY_STARTED = "utilization_recovery_started"  # Work injection started
    UTILIZATION_RECOVERY_COMPLETED = "utilization_recovery_completed"  # Work injection completed
    UTILIZATION_RECOVERY_FAILED = "utilization_recovery_failed"  # Work injection failed

    # Jan 4, 2026: Fast failure detection (Phase 4 - P2P resilience)
    FAST_FAILURE_ALERT = "fast_failure_alert"  # 10-minute failure detected
    FAST_FAILURE_RECOVERY = "fast_failure_recovery"  # 30-minute escalation triggered
    FAST_FAILURE_RECOVERED = "fast_failure_recovered"  # Cluster returned to healthy state

    # Jan 4, 2026: Cluster resilience orchestration (Phase 6 - P2P resilience)
    RESILIENCE_ESCALATION = "resilience_escalation"  # Coordinated escalation across components

    # Cluster status events
    CLUSTER_STATUS_CHANGED = "cluster_status_changed"
    CLUSTER_STALL_DETECTED = "cluster_stall_detected"  # Dec 2025: Node(s) stuck with no game progress
    NODE_UNHEALTHY = "node_unhealthy"
    NODE_RECOVERED = "node_recovered"
    NODE_SUSPECT = "node_suspect"  # Dec 2025: Node entering SUSPECT state (grace period before DEAD)
    NODE_RETIRED = "node_retired"  # Dec 2025: Node explicitly retired from cluster
    NODE_ACTIVATED = "node_activated"  # Node activated by cluster activator
    NODE_TERMINATED = "node_terminated"  # Node terminated/deprovisioned
    NODE_INCOMPATIBLE_WITH_WORKLOAD = "node_incompatible_with_workload"  # Dec 2025: Node has no compatible selfplay configs

    # SSH Liveness events (December 30, 2025)
    SSH_LIVENESS_CHECK_STARTED = "ssh_liveness_check_started"  # SSH liveness check initiated
    SSH_LIVENESS_CHECK_SUCCEEDED = "ssh_liveness_check_succeeded"  # Node responded via SSH
    SSH_LIVENESS_CHECK_FAILED = "ssh_liveness_check_failed"  # SSH check failed (timeout/refused/auth)
    SSH_NODE_UNRESPONSIVE = "ssh_node_unresponsive"  # Node unresponsive via SSH after retries
    SSH_NODE_RECOVERED = "ssh_node_recovered"  # Previously unresponsive node now responds via SSH

    # Circuit Breaker events (January 2026 Sprint 10)
    CIRCUIT_BREAKER_OPENED = "circuit_breaker_opened"  # Node circuit breaker opened (too many failures)
    CIRCUIT_BREAKER_CLOSED = "circuit_breaker_closed"  # Node circuit breaker closed (recovered)
    CIRCUIT_BREAKER_HALF_OPEN = "circuit_breaker_half_open"  # Circuit testing recovery
    CIRCUIT_BREAKER_THRESHOLD = "circuit_breaker_threshold"  # Too many open circuits (>20% of nodes)
    ESCALATION_TIER_CHANGED = "escalation_tier_changed"  # Circuit breaker escalation tier changed (Sprint 12)
    CIRCUIT_RESET = "circuit_reset"  # Circuit breaker proactively reset after peer recovery (Sprint 12 Session 8)

    # Lock/Synchronization events (December 2025)
    LOCK_TIMEOUT = "lock_timeout"
    DEADLOCK_DETECTED = "deadlock_detected"

    # Checkpoint events (December 2025)
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_LOADED = "checkpoint_loaded"

    # Backup events (December 2025)
    DATA_BACKUP_COMPLETED = "data_backup_completed"  # External drive backup finished
    S3_BACKUP_COMPLETED = "s3_backup_completed"  # S3 backup finished (after model promotion)

    # CPU Pipeline events (December 2025)
    CPU_PIPELINE_JOB_COMPLETED = "cpu_pipeline_job_completed"  # Vast CPU job finished

    # Task lifecycle events (December 2025)
    TASK_SPAWNED = "task_spawned"
    TASK_HEARTBEAT = "task_heartbeat"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_ORPHANED = "task_orphaned"
    TASK_CANCELLED = "task_cancelled"

    # Job spawn verification events (January 2026 - Sprint 6)
    # Used by SelfplayScheduler to track spawn success/failure for capacity estimation
    JOB_SPAWN_VERIFIED = "job_spawn_verified"  # Job confirmed running within timeout
    JOB_SPAWN_FAILED = "job_spawn_failed"  # Job not confirmed running, spawn failed

    # Capacity/Resource events (December 2025)
    CLUSTER_CAPACITY_CHANGED = "cluster_capacity_changed"
    NODE_CAPACITY_UPDATED = "node_capacity_updated"
    BACKPRESSURE_ACTIVATED = "backpressure_activated"
    BACKPRESSURE_RELEASED = "backpressure_released"
    IDLE_RESOURCE_DETECTED = "idle_resource_detected"  # Phase 21.2: Idle GPU/CPU detected
    CLUSTER_UNDERUTILIZED = "cluster_underutilized"  # Dec 30, 2025: Many GPUs idle, needs remediation
    CLUSTER_UTILIZATION_RECOVERED = "cluster_utilization_recovered"  # Dec 30, 2025: Utilization back to healthy

    # Availability/Provisioning events (December 28, 2025)
    CAPACITY_LOW = "capacity_low"  # Cluster GPU capacity below threshold
    CAPACITY_RESTORED = "capacity_restored"  # Capacity back above threshold
    NODE_PROVISIONED = "node_provisioned"  # New node created via provider API
    NODE_PROVISION_FAILED = "node_provision_failed"  # Failed to provision node
    BUDGET_EXCEEDED = "budget_exceeded"  # Hourly/daily budget exceeded
    BUDGET_ALERT = "budget_alert"  # Approaching budget threshold

    # Promotion lifecycle events (December 2025)
    PROMOTION_ROLLED_BACK = "promotion_rolled_back"

    # Quality feedback events (December 2025)
    PARITY_FAILURE_RATE_CHANGED = "parity_failure_rate_changed"

    # Leader election events (December 2025)
    LEADER_ELECTED = "leader_elected"
    LEADER_LOST = "leader_lost"
    LEADER_STEPDOWN = "leader_stepdown"
    SPLIT_BRAIN_DETECTED = "split_brain_detected"  # Multiple leaders detected in cluster
    SPLIT_BRAIN_RESOLVED = "split_brain_resolved"  # Split-brain resolved (non-canonical leaders demoted)
    LEADER_HEARTBEAT_MISSING = "leader_heartbeat_missing"  # P0 Dec 2025: Leader heartbeat delayed (monitoring)
    LEADER_LEASE_EXPIRED = "leader_lease_expired"  # Jan 2026: Leader lease expired without stepdown

    # P2P State persistence events (December 2025)
    STATE_PERSISTED = "state_persisted"  # P2P state saved to database
    EPOCH_ADVANCED = "epoch_advanced"  # Cluster epoch incremented (leader change)

    # Encoding/Processing events (December 2025)

    # Error Recovery & Resilience events (December 2025)
    TRAINING_ROLLBACK_NEEDED = "training_rollback_needed"  # Rollback to previous checkpoint
    TRAINING_ROLLBACK_COMPLETED = "training_rollback_completed"
    MODEL_CORRUPTED = "model_corrupted"  # Model file corruption detected
    COORDINATOR_HEALTHY = "coordinator_healthy"  # Coordinator healthy signal
    COORDINATOR_UNHEALTHY = "coordinator_unhealthy"  # Coordinator unhealthy signal
    COORDINATOR_HEALTH_DEGRADED = "coordinator_health_degraded"  # Coordinator not fully healthy
    # Session 16 (Jan 2026): Coordinator failover for cluster resilience
    COORDINATOR_FAILOVER = "coordinator_failover"  # Standby taking over as primary
    COORDINATOR_HANDOFF = "coordinator_handoff"  # Primary handing off to recovered primary
    COORDINATOR_EMERGENCY_SHUTDOWN = "coordinator_emergency_shutdown"  # Emergency shutdown (memory/OOM)
    COORDINATOR_SHUTDOWN = "coordinator_shutdown"  # Graceful coordinator shutdown
    COORDINATOR_INIT_FAILED = "coordinator_init_failed"  # Coordinator failed to initialize
    HANDLER_TIMEOUT = "handler_timeout"  # Event handler timed out
    HANDLER_FAILED = "handler_failed"  # Event handler threw exception
    TASK_ABANDONED = "task_abandoned"  # Task intentionally abandoned (not orphaned)
    RESOURCE_CONSTRAINT_DETECTED = "resource_constraint_detected"  # Resource limit hit

    # Coordinator heartbeat events (December 2025)
    COORDINATOR_HEARTBEAT = "coordinator_heartbeat"  # Liveness signal from coordinator

    # Phase 3 feedback loop events (December 2025)
    # Note: SYNC_STALLED, NODE_OVERLOADED, TRAINING_LOSS_TREND already defined above
    TRAINING_EARLY_STOPPED = "training_early_stopped"  # Early stopping triggered (stagnation/regression)

    # Cluster-wide idle state broadcast events (December 2025)
    IDLE_STATE_BROADCAST = "idle_state_broadcast"  # Node broadcasting its idle state to cluster
    IDLE_STATE_REQUEST = "idle_state_request"  # Request all nodes to broadcast idle state

    # Disk space management events (December 2025)
    DISK_SPACE_LOW = "disk_space_low"  # Disk usage above threshold (warning/critical)
    DISK_CLEANUP_TRIGGERED = "disk_cleanup_triggered"  # Proactive cleanup started/completed

    # Dead Letter Queue events (December 29, 2025)
    DLQ_STALE_EVENTS = "dlq_stale_events"  # Stale events detected in DLQ
    DLQ_EVENTS_REPLAYED = "dlq_events_replayed"  # DLQ events replayed successfully
    DLQ_EVENTS_PURGED = "dlq_events_purged"  # DLQ events purged (too old/invalid)

    # Comprehensive Model Evaluation events (January 2026)
    # Ensure all models are evaluated under all compatible harnesses
    COMPREHENSIVE_EVALUATION_STARTED = "comprehensive_evaluation_started"  # 6-hour evaluation cycle started
    COMPREHENSIVE_EVALUATION_COMPLETED = "comprehensive_evaluation_completed"  # Evaluation cycle completed
    MODEL_HARNESS_EVALUATED = "model_harness_evaluated"  # Single (model, harness, config) combo evaluated
    CLUSTER_MODEL_INVENTORY_UPDATED = "cluster_model_inventory_updated"  # Model list across cluster refreshed
    UNEVALUATED_MODELS_DETECTED = "unevaluated_models_detected"  # Models without ratings discovered

    # Tournament Data Pipeline events (January 2026)
    # Automatically feed tournament/gauntlet games into training
    TOURNAMENT_DATA_READY = "tournament_data_ready"  # Tournament NPZ exported and ready for training
    TOURNAMENT_PIPELINE_COMPLETED = "tournament_pipeline_completed"  # Pipeline cycle completed
    TOURNAMENT_TRAINING_TRIGGERED = "tournament_training_triggered"  # Training triggered on tournament data


@dataclass
class DataEvent:
    """A data pipeline event."""

    event_type: DataEventType
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # Component that generated the event

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataEvent:
        """Create from dictionary."""
        return cls(
            event_type=DataEventType(data["event_type"]),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            source=data.get("source", ""),
        )


EventCallback = Callable[[DataEvent], Union[None, asyncio.Future]]


class EventBus:
    """Async event bus for component coordination.

    Supports both sync and async callbacks. Events are delivered
    in order of subscription.

    December 2025: Added subscription registry with warnings for unsubscribed events.
    """

    def __init__(self, max_history: int = 1000, warn_unsubscribed: bool = True):
        self._subscribers: dict[DataEventType, list[EventCallback]] = {}
        self._global_subscribers: list[EventCallback] = []
        self._event_history: list[DataEvent] = []
        self._max_history = max_history
        self._lock = asyncio.Lock()

        # Subscription registry (December 2025)
        self._warn_unsubscribed = warn_unsubscribed
        self._published_event_types: dict[DataEventType, int] = {}  # type -> count
        self._warned_event_types: set = set()  # Types we've already warned about

        # Observability metrics (December 2025)
        self._start_time = time.time()
        self._total_events_published = 0
        self._total_callbacks_invoked = 0
        self._total_callback_errors = 0
        self._callback_latencies: list[float] = []  # Recent latencies in ms
        self._max_latency_samples = 1000
        self._errors_by_type: dict[DataEventType, int] = {}
        self._last_event_time: float = 0.0

    def subscribe(
        self,
        event_type: DataEventType | None,
        callback: EventCallback,
    ) -> None:
        """Subscribe to events.

        Args:
            event_type: Specific event type, or None for all events
            callback: Function to call when event occurs. Can be sync or async.
        """
        if event_type is None:
            self._global_subscribers.append(callback)
        else:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def unsubscribe(
        self,
        event_type: DataEventType | None,
        callback: EventCallback,
    ) -> bool:
        """Unsubscribe from events.

        Returns True if callback was found and removed.
        """
        if event_type is None:
            if callback in self._global_subscribers:
                self._global_subscribers.remove(callback)
                return True
        else:
            if event_type in self._subscribers and callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                return True
        return False

    async def publish(self, event: DataEvent, bridge_cross_process: bool = True) -> None:
        """Publish an event to all subscribers.

        Callbacks are invoked in order. Async callbacks are awaited.
        Errors in callbacks are logged but don't prevent delivery to
        other subscribers.

        Args:
            event: The event to publish
            bridge_cross_process: If True, also bridge to cross-process queue
        """
        # Dec 31, 2025: Defensive handling for bare DataEventType passed instead of DataEvent
        # This fixes AttributeError when callers pass DataEventType enum directly
        if isinstance(event, DataEventType):
            event = DataEvent(event_type=event, payload={}, source="unknown")

        async with self._lock:
            # Store in history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

            # Track published event types (December 2025)
            self._published_event_types[event.event_type] = (
                self._published_event_types.get(event.event_type, 0) + 1
            )

        # Bridge to cross-process queue for multi-daemon coordination
        if bridge_cross_process:
            _bridge_to_cross_process(event)

        # Get all callbacks for this event
        callbacks = list(self._global_subscribers)
        if event.event_type in self._subscribers:
            callbacks.extend(self._subscribers[event.event_type])

        # Warn if no subscribers (December 2025)
        if (
            self._warn_unsubscribed
            and not callbacks
            and event.event_type not in self._warned_event_types
        ):
            self._warned_event_types.add(event.event_type)
            print(
                f"[EventBus] WARNING: Event {event.event_type.value} published "
                f"but has no subscribers. Consider adding a handler."
            )

        # Invoke each callback with latency tracking (December 2025)
        for callback in callbacks:
            self._total_callbacks_invoked += 1
            callback_start = time.time()
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                self._total_callback_errors += 1
                self._errors_by_type[event.event_type] = (
                    self._errors_by_type.get(event.event_type, 0) + 1
                )
                print(f"[EventBus] Error in subscriber for {event.event_type.value}: {e}")

                # Capture to dead letter queue if enabled (December 2025)
                if hasattr(self, "_dlq") and self._dlq is not None:
                    try:
                        self._dlq.capture(
                            event_type=event.event_type.value,
                            payload=event.payload,
                            handler_name=getattr(callback, "__name__", "unknown"),
                            error=str(e),
                            source="data_events",
                        )
                    except Exception as dlq_error:
                        print(f"[EventBus] Failed to capture to DLQ: {dlq_error}")
            finally:
                # Track callback latency
                latency_ms = (time.time() - callback_start) * 1000
                self._callback_latencies.append(latency_ms)
                if len(self._callback_latencies) > self._max_latency_samples:
                    self._callback_latencies = self._callback_latencies[-self._max_latency_samples:]

        # Update event metrics
        self._total_events_published += 1
        self._last_event_time = time.time()

    def publish_sync(self, event: DataEvent, bridge_cross_process: bool = True) -> None:
        """Publish an event synchronously (non-async context).

        Creates a new event loop if needed. Use publish() when possible.
        """
        # Bridge immediately in sync context (doesn't need async)
        if bridge_cross_process:
            _bridge_to_cross_process(event)

        try:
            loop = asyncio.get_running_loop()
            # Don't bridge again in the async publish
            loop.create_task(self.publish(event, bridge_cross_process=False))
        except RuntimeError:
            # No running loop - run synchronously
            asyncio.run(self.publish(event, bridge_cross_process=False))

    def emit(
        self,
        event_type: DataEventType,
        payload: dict[str, Any] | None = None,
        source: str = "",
        bridge_cross_process: bool = True,
    ) -> None:
        """Convenience method to emit an event with type and payload.

        This is an alias for publish() that accepts event type and payload
        separately, creating a DataEvent internally. It provides API
        consistency with other event systems.

        December 28, 2025: Added to fix API mismatch where code calls
        bus.emit(DataEventType, payload) but EventBus only had publish(DataEvent).

        Args:
            event_type: The event type to emit
            payload: Event payload dictionary
            source: Component that generated the event
            bridge_cross_process: If True, also bridge to cross-process queue

        Usage:
            bus = get_event_bus()
            bus.emit(DataEventType.TRAINING_COMPLETED, {"config_key": "hex8_2p"})
        """
        event = DataEvent(
            event_type=event_type,
            payload=payload or {},
            source=source,
        )
        # Use sync version since emit() is called from sync contexts
        self.publish_sync(event, bridge_cross_process=bridge_cross_process)

    async def emit_async(
        self,
        event_type: DataEventType,
        payload: dict[str, Any] | None = None,
        source: str = "",
        bridge_cross_process: bool = True,
    ) -> None:
        """Async version of emit() for use in async contexts.

        December 28, 2025: Added for async callers.

        Args:
            event_type: The event type to emit
            payload: Event payload dictionary
            source: Component that generated the event
            bridge_cross_process: If True, also bridge to cross-process queue
        """
        event = DataEvent(
            event_type=event_type,
            payload=payload or {},
            source=source,
        )
        await self.publish(event, bridge_cross_process=bridge_cross_process)

    def get_history(
        self,
        event_type: DataEventType | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[DataEvent]:
        """Get recent events from history.

        Args:
            event_type: Filter by event type (None for all)
            since: Only events after this timestamp
            limit: Maximum number of events to return
        """
        events = self._event_history

        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]

        if since is not None:
            events = [e for e in events if e.timestamp > since]

        return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history = []

    # =========================================================================
    # Subscription Registry (December 2025)
    # =========================================================================

    def get_subscribed_event_types(self) -> list[DataEventType]:
        """Get list of event types that have at least one subscriber."""
        return list(self._subscribers.keys())

    def get_unsubscribed_published_types(self) -> list[DataEventType]:
        """Get event types that have been published but have no subscribers.

        This is useful for debugging to find events that are being published
        but nobody is listening to.
        """
        unsubscribed = []
        for event_type in self._published_event_types:
            has_specific = event_type in self._subscribers and self._subscribers[event_type]
            has_global = bool(self._global_subscribers)
            if not has_specific and not has_global:
                unsubscribed.append(event_type)
        return unsubscribed

    def get_subscriber_count(self, event_type: DataEventType) -> int:
        """Get the number of subscribers for a specific event type."""
        count = len(self._global_subscribers)
        if event_type in self._subscribers:
            count += len(self._subscribers[event_type])
        return count

    def get_subscription_stats(self) -> dict[str, Any]:
        """Get statistics about subscriptions and published events.

        Returns:
            Dict with subscription statistics
        """
        return {
            "subscribed_types": [t.value for t in self.get_subscribed_event_types()],
            "published_types": {t.value: c for t, c in self._published_event_types.items()},
            "unsubscribed_published": [t.value for t in self.get_unsubscribed_published_types()],
            "global_subscribers": len(self._global_subscribers),
            "warned_types": [t.value for t in self._warned_event_types],
            "total_events_published": sum(self._published_event_types.values()),
        }

    # =========================================================================
    # Observability Metrics (December 2025)
    # =========================================================================

    def get_observability_metrics(self) -> dict[str, Any]:
        """Get comprehensive observability metrics for the event bus.

        Returns:
            Dict with metrics including:
            - Throughput (events/callbacks per second)
            - Latency statistics (mean, p50, p95, p99)
            - Error rates
            - Uptime and activity
        """
        import statistics as stats

        uptime = time.time() - self._start_time
        events_per_second = self._total_events_published / uptime if uptime > 0 else 0.0

        # Calculate latency percentiles
        latency_stats = {
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
        if self._callback_latencies:
            sorted_latencies = sorted(self._callback_latencies)
            n = len(sorted_latencies)
            latency_stats = {
                "mean_ms": round(stats.mean(sorted_latencies), 2),
                "p50_ms": round(sorted_latencies[int(n * 0.5)], 2),
                "p95_ms": round(sorted_latencies[int(n * 0.95)], 2),
                "p99_ms": round(sorted_latencies[int(n * 0.99)], 2),
                "max_ms": round(max(sorted_latencies), 2),
            }

        # Error rate
        error_rate = (
            self._total_callback_errors / self._total_callbacks_invoked
            if self._total_callbacks_invoked > 0 else 0.0
        )

        return {
            "uptime_seconds": round(uptime, 1),
            "total_events_published": self._total_events_published,
            "total_callbacks_invoked": self._total_callbacks_invoked,
            "total_callback_errors": self._total_callback_errors,
            "error_rate": round(error_rate, 4),
            "events_per_second": round(events_per_second, 2),
            "latency": latency_stats,
            "errors_by_type": {t.value: c for t, c in self._errors_by_type.items()},
            "last_event_time": self._last_event_time,
            "seconds_since_last_event": round(time.time() - self._last_event_time, 1) if self._last_event_time else None,
            "subscriber_counts": {
                t.value: len(subs) for t, subs in self._subscribers.items()
            },
            "history_size": len(self._event_history),
        }

    def get_health_status(self) -> dict[str, Any]:
        """Get health status of the event bus.

        Returns:
            Dict with health indicators
        """
        metrics = self.get_observability_metrics()

        # Determine health score
        health_score = 1.0
        issues = []

        # Check error rate
        if metrics["error_rate"] > 0.1:
            health_score -= 0.3
            issues.append(f"High error rate: {metrics['error_rate']:.1%}")
        elif metrics["error_rate"] > 0.05:
            health_score -= 0.1
            issues.append(f"Elevated error rate: {metrics['error_rate']:.1%}")

        # Check latency
        if metrics["latency"]["p95_ms"] > 1000:
            health_score -= 0.2
            issues.append(f"High p95 latency: {metrics['latency']['p95_ms']}ms")
        elif metrics["latency"]["p95_ms"] > 500:
            health_score -= 0.1
            issues.append(f"Elevated p95 latency: {metrics['latency']['p95_ms']}ms")

        # Check for stale bus
        if metrics["seconds_since_last_event"] and metrics["seconds_since_last_event"] > 300:
            health_score -= 0.1
            issues.append(f"No events for {metrics['seconds_since_last_event']}s")

        # Determine status
        if health_score >= 0.9:
            status = "healthy"
        elif health_score >= 0.7:
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "health_score": round(health_score, 2),
            "issues": issues,
            "metrics_summary": {
                "events_published": metrics["total_events_published"],
                "error_rate": metrics["error_rate"],
                "p95_latency_ms": metrics["latency"]["p95_ms"],
                "events_per_second": metrics["events_per_second"],
            },
        }

    def reset_metrics(self) -> None:
        """Reset observability metrics (for testing)."""
        self._total_events_published = 0
        self._total_callbacks_invoked = 0
        self._total_callback_errors = 0
        self._callback_latencies.clear()
        self._errors_by_type.clear()
        self._last_event_time = 0.0
        self._start_time = time.time()

    def has_subscribers(self, event_type: DataEventType) -> bool:
        """Check if an event type has any subscribers (specific or global)."""
        return self.get_subscriber_count(event_type) > 0

    def health_check(self) -> "HealthCheckResult":
        """Return health status for daemon monitoring.

        December 2025: Added for DaemonManager integration.

        Health criteria:
        - Error rate < 10% (healthy) or < 20% (degraded)
        - p95 latency < 1000ms (healthy) or < 2000ms (degraded)
        - Should have processed at least some events

        Returns:
            HealthCheckResult compatible with DaemonManager
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        # Get existing health status
        health_status = self.get_health_status()
        metrics = self.get_observability_metrics()

        # Map status to HealthCheckResult
        is_healthy = health_status["status"] == "healthy"
        is_degraded = health_status["status"] == "degraded"

        # Build detailed status
        details = {
            "health_score": health_status["health_score"],
            "total_events_published": metrics["total_events_published"],
            "total_callbacks_invoked": metrics["total_callbacks_invoked"],
            "total_callback_errors": metrics["total_callback_errors"],
            "error_rate": metrics["error_rate"],
            "p95_latency_ms": metrics["latency"]["p95_ms"],
            "events_per_second": metrics["events_per_second"],
            "subscriber_count": sum(
                len(subs) for subs in self._subscribers.values()
            ) + len(self._global_subscribers),
            "subscribed_types": len(self._subscribers),
            "uptime_seconds": metrics["uptime_seconds"],
            "issues": health_status.get("issues", []),
        }

        if is_healthy:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"EventBus healthy: {metrics['total_events_published']} events, {metrics['error_rate']:.1%} error rate",
                details=details,
            )
        elif is_degraded:
            return HealthCheckResult(
                healthy=True,  # Degraded is still operational
                status=CoordinatorStatus.DEGRADED,
                message=f"EventBus degraded: {'; '.join(health_status.get('issues', []))}",
                details=details,
            )
        else:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.UNHEALTHY,
                message=f"EventBus unhealthy: {'; '.join(health_status.get('issues', []))}",
                details=details,
            )


def get_event_bus() -> EventBus:
    """Get the global event bus singleton.

    .. deprecated:: December 2025
        Use :func:`app.coordination.event_router.get_router` instead for unified
        event routing across all event systems.
    """
    import warnings
    warnings.warn(
        "get_event_bus() from data_events is deprecated. "
        "Use get_router() from app.coordination.event_router instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def reset_event_bus() -> None:
    """Reset the global event bus (for testing)."""
    global _event_bus
    _event_bus = None


# Cross-process bridging configuration
# Events that should be propagated to cross-process queue for multi-daemon coordination
CROSS_PROCESS_EVENT_TYPES = {
    # Success events - coordination across processes
    DataEventType.MODEL_PROMOTED,
    DataEventType.TIER_PROMOTION,  # Difficulty ladder progression
    DataEventType.TRAINING_STARTED,
    DataEventType.TRAINING_COMPLETED,
    DataEventType.EVALUATION_COMPLETED,
    DataEventType.CURRICULUM_REBALANCED,
    DataEventType.CURRICULUM_ADVANCED,  # Curriculum tier progression
    DataEventType.CURRICULUM_ROLLBACK_COMPLETED,  # Sprint 16.1: Rollback confirmation
    DataEventType.SELFPLAY_TARGET_UPDATED,  # Dynamic selfplay scaling
    DataEventType.ELO_SIGNIFICANT_CHANGE,
    DataEventType.P2P_MODEL_SYNCED,
    DataEventType.PLATEAU_DETECTED,
    DataEventType.DATA_SYNC_COMPLETED,
    DataEventType.HYPERPARAMETER_UPDATED,
    DataEventType.GAME_SYNCED,  # Ephemeral sync events
    DataEventType.DATA_STALE,  # Training data freshness
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
    DataEventType.DAEMON_STATUS_CHANGED,  # Watchdog alerts for daemon health
    DataEventType.DAEMON_PERMANENTLY_FAILED,  # Dec 2025: Exceeded hourly restart limit
    DataEventType.DAEMON_CRASH_LOOP_DETECTED,  # Dec 2025: Early warning for crash loops
    # Trigger events - distributed optimization
    DataEventType.CMAES_TRIGGERED,
    DataEventType.NAS_TRIGGERED,
    DataEventType.TRAINING_THRESHOLD_REACHED,
    DataEventType.CACHE_INVALIDATED,
    # Regression events - unified detection across all processes
    DataEventType.REGRESSION_DETECTED,
    DataEventType.REGRESSION_SEVERE,
    DataEventType.REGRESSION_CRITICAL,
    DataEventType.REGRESSION_CLEARED,
}


def _bridge_to_cross_process(event: DataEvent) -> None:
    """Bridge event to cross-process queue if it's a cross-process event type.

    This allows events published to the in-memory EventBus to also be
    visible to other daemon processes via the SQLite-backed queue.
    """
    if event.event_type not in CROSS_PROCESS_EVENT_TYPES:
        return

    try:
        # Phase 9 (Dec 2025): Import directly from cross_process_events to avoid
        # circular import with event_router (which imports from this module)
        from app.coordination.cross_process_events import bridge_to_cross_process
        bridge_to_cross_process(event.event_type.value, event.payload, event.source)
    except Exception as e:
        # Don't fail the main event if cross-process bridging fails
        print(f"[EventBus] Cross-process bridge failed: {e}")


# Convenience functions for common events


async def emit_data_event(
    event_type: DataEventType | str,
    payload: dict[str, Any] | None = None,
    source: str = "",
    *,
    bridge_cross_process: bool = True,
) -> None:
    """Emit an arbitrary DataEvent.

    This is a thin helper used by the unified router and by quality/monitoring
    utilities that want to publish DataEventType values without depending on
    the full EventBus API.

    Args:
        event_type: Event type enum or its string value
        payload: Event payload
        source: Component that generated the event
        bridge_cross_process: Whether to bridge to cross-process queue
    """
    if isinstance(event_type, str):
        event_type = DataEventType(event_type)

    await get_event_bus().publish(
        DataEvent(
            event_type=event_type,
            payload=payload or {},
            source=source,
        ),
        bridge_cross_process=bridge_cross_process,
    )


async def emit_new_games(host: str, new_games: int, total_games: int, source: str = "") -> None:
    """Emit a NEW_GAMES_AVAILABLE event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NEW_GAMES_AVAILABLE,
        payload={
            "host": host,
            "new_games": new_games,
            "total_games": total_games,
        },
        source=source,
    ))


async def emit_training_threshold(config: str, games: int, source: str = "") -> None:
    """Emit a TRAINING_THRESHOLD_REACHED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_THRESHOLD_REACHED,
        payload={
            "config": config,
            "games": games,
        },
        source=source,
    ))


async def emit_training_completed(
    config: str,
    success: bool,
    duration: float,
    model_path: str | None = None,
    source: str = "",
) -> None:
    """Emit a TRAINING_COMPLETED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_COMPLETED,
        payload={
            "config": config,
            "success": success,
            "duration": duration,
            "model_path": model_path,
        },
        source=source,
    ))


async def emit_evaluation_completed(
    config: str,
    elo: float,
    games_played: int,
    win_rate: float,
    source: str = "",
) -> None:
    """Emit an EVALUATION_COMPLETED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.EVALUATION_COMPLETED,
        payload={
            "config": config,
            "elo": elo,
            "games_played": games_played,
            "win_rate": win_rate,
        },
        source=source,
    ))


async def emit_model_promoted(
    model_id: str,
    config: str,
    elo: float,
    elo_gain: float,
    source: str = "",
) -> None:
    """Emit a MODEL_PROMOTED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.MODEL_PROMOTED,
        payload={
            "model_id": model_id,
            "config": config,
            "elo": elo,
            "elo_gain": elo_gain,
        },
        source=source,
    ))


async def emit_error(
    component: str,
    error: str,
    details: dict[str, Any] | None = None,
    source: str = "",
) -> None:
    """Emit an ERROR event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.ERROR,
        payload={
            "component": component,
            "error": error,
            "details": details or {},
        },
        source=source,
    ))


async def emit_elo_updated(
    config: str,
    model_id: str,
    new_elo: float,
    old_elo: float,
    games_played: int,
    source: str = "",
) -> None:
    """Emit an ELO_UPDATED event and check for significant changes.

    If the Elo change exceeds the threshold from unified config,
    also emits an ELO_SIGNIFICANT_CHANGE event to trigger curriculum rebalancing.
    """
    elo_change = new_elo - old_elo

    # Emit the standard Elo update event
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.ELO_UPDATED,
        payload={
            "config": config,
            "model_id": model_id,
            "new_elo": new_elo,
            "old_elo": old_elo,
            "elo_change": elo_change,
            "games_played": games_played,
        },
        source=source,
    ))

    # Check for significant change (threshold from unified config)
    try:
        from app.config.unified_config import get_config
        config_obj = get_config()
        threshold = config_obj.curriculum.elo_change_threshold
        should_rebalance = config_obj.curriculum.rebalance_on_elo_change
    except ImportError:
        threshold = 20  # December 2025: Lowered from 50 to enable faster curriculum adaptation
        should_rebalance = True

    if should_rebalance and abs(elo_change) >= threshold:
        await get_event_bus().publish(DataEvent(
            event_type=DataEventType.ELO_SIGNIFICANT_CHANGE,
            payload={
                "config": config,
                "model_id": model_id,
                "elo_change": elo_change,
                "new_elo": new_elo,
                "threshold": threshold,
            },
            source=source,
        ))


async def emit_elo_velocity_changed(
    config_key: str,
    velocity: float,
    previous_velocity: float,
    trend: str,
    source: str = "queue_populator",
) -> None:
    """Emit an ELO_VELOCITY_CHANGED event.

    P10-LOOP-3 (Dec 2025): Signals when Elo improvement velocity changes significantly.
    This triggers selfplay rate adjustments:
    - Accelerating velocity → increase selfplay to capitalize on momentum
    - Decelerating velocity → reduce selfplay, focus on quality
    - Negative velocity → boost exploration to escape local minimum

    Args:
        config_key: Configuration key (e.g., "hex8_2p")
        velocity: Current Elo points per day
        previous_velocity: Previous velocity measurement
        trend: Velocity trend ("accelerating", "stable", "decelerating")
        source: Component emitting the event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.ELO_VELOCITY_CHANGED,
        payload={
            "config_key": config_key,
            "velocity": velocity,
            "previous_velocity": previous_velocity,
            "velocity_change": velocity - previous_velocity,
            "trend": trend,
        },
        source=source,
    ))


async def emit_quality_score_updated(
    game_id: str,
    quality_score: float,
    quality_category: str,
    training_weight: float,
    game_length: int = 0,
    is_decisive: bool = False,
    source: str = "",
) -> None:
    """Emit a QUALITY_SCORE_UPDATED event.

    Args:
        game_id: Unique game identifier
        quality_score: Computed quality score (0-1)
        quality_category: Category (excellent/good/adequate/poor/unusable)
        training_weight: Weight for training sample selection
        game_length: Number of moves in the game
        is_decisive: Whether game had a clear winner
        source: Component that computed the quality
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.QUALITY_SCORE_UPDATED,
        payload={
            "game_id": game_id,
            "quality_score": quality_score,
            "quality_category": quality_category,
            "training_weight": training_weight,
            "game_length": game_length,
            "is_decisive": is_decisive,
        },
        source=source,
    ))


async def emit_quality_degraded(
    config_key: str,
    quality_score: float,
    threshold: float = 0.6,
    previous_score: float = 0.0,
    source: str = "quality_monitor",
) -> None:
    """Emit a QUALITY_DEGRADED event when quality drops below threshold.

    Phase 5 (Dec 2025): Enables feedback loop to reduce selfplay allocation
    for configs producing low-quality games.

    Args:
        config_key: Board configuration (e.g., "hex8_2p")
        quality_score: Current quality score (0-1)
        threshold: Threshold that was crossed
        previous_score: Previous quality score for comparison
        source: Component that detected the degradation
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.QUALITY_DEGRADED,
        payload={
            "config_key": config_key,
            "quality_score": quality_score,
            "threshold": threshold,
            "previous_score": previous_score,
            "degradation_delta": previous_score - quality_score if previous_score else 0.0,
        },
        source=source,
    ))


async def emit_quality_check_requested(
    config_key: str,
    reason: str,
    source: str = "feedback_loop_controller",
    priority: str = "normal",
) -> None:
    """Emit a QUALITY_CHECK_REQUESTED event to trigger on-demand quality check.

    Phase 9 (Dec 2025): Closes the feedback loop between training loss anomalies
    and data quality verification. When training loss spikes, this requests the
    QualityMonitorDaemon to perform an immediate quality check.

    Args:
        config_key: Board configuration (e.g., "hex8_2p")
        reason: Reason for the quality check request
        source: Component requesting the check
        priority: Priority level ("normal", "high", "urgent")
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.QUALITY_CHECK_REQUESTED,
        payload={
            "config_key": config_key,
            "reason": reason,
            "priority": priority,
        },
        source=source,
    ))


async def emit_curriculum_rebalanced(
    config: str,
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    trigger: str = "scheduled",
    source: str = "",
) -> None:
    """Emit a CURRICULUM_REBALANCED event.

    Args:
        config: Board configuration (e.g., "square8_2p")
        old_weights: Previous curriculum weights
        new_weights: New curriculum weights
        trigger: What triggered the rebalance ("scheduled", "elo_change", "manual")
        source: Component that triggered the event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CURRICULUM_REBALANCED,
        payload={
            "config": config,
            "old_weights": old_weights,
            "new_weights": new_weights,
            "trigger": trigger,
        },
        source=source,
    ))


async def emit_plateau_detected(
    config: str,
    current_elo: float,
    plateau_duration_games: int,
    plateau_duration_seconds: float,
    source: str = "",
) -> None:
    """Emit a PLATEAU_DETECTED event.

    Args:
        config: Board configuration
        current_elo: Current Elo rating
        plateau_duration_games: Number of games in plateau
        plateau_duration_seconds: Time duration of plateau
        source: Component that detected the plateau
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.PLATEAU_DETECTED,
        payload={
            "config": config,
            "current_elo": current_elo,
            "plateau_duration_games": plateau_duration_games,
            "plateau_duration_seconds": plateau_duration_seconds,
        },
        source=source,
    ))


async def emit_cmaes_triggered(
    config: str,
    reason: str,
    current_params: dict[str, Any],
    source: str = "",
) -> None:
    """Emit a CMAES_TRIGGERED event.

    Args:
        config: Board configuration
        reason: Why CMA-ES was triggered (e.g., "plateau_detected", "manual")
        current_params: Current hyperparameters before optimization
        source: Component that triggered CMA-ES
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CMAES_TRIGGERED,
        payload={
            "config": config,
            "reason": reason,
            "current_params": current_params,
        },
        source=source,
    ))


async def emit_nas_triggered(
    config: str,
    reason: str,
    search_space: dict[str, Any] | None = None,
    source: str = "",
) -> None:
    """Emit a NAS_TRIGGERED event.

    Args:
        config: Board configuration
        reason: Why NAS was triggered
        search_space: Optional architecture search space configuration
        source: Component that triggered NAS
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NAS_TRIGGERED,
        payload={
            "config": config,
            "reason": reason,
            "search_space": search_space or {},
        },
        source=source,
    ))


async def emit_hyperparameter_updated(
    config: str,
    param_name: str,
    old_value: Any,
    new_value: Any,
    optimizer: str = "manual",
    source: str = "",
) -> None:
    """Emit a HYPERPARAMETER_UPDATED event.

    Args:
        config: Board configuration
        param_name: Name of the parameter that changed
        old_value: Previous value
        new_value: New value
        optimizer: What triggered the update ("cmaes", "nas", "manual")
        source: Component that updated the parameter
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.HYPERPARAMETER_UPDATED,
        payload={
            "config": config,
            "param_name": param_name,
            "old_value": old_value,
            "new_value": new_value,
            "optimizer": optimizer,
        },
        source=source,
    ))


async def emit_exploration_boost(
    config_key: str,
    boost_factor: float,
    reason: str,
    anomaly_count: int = 0,
    source: str = "feedback_loop_controller",
) -> None:
    """Emit an EXPLORATION_BOOST event.

    P11-CRITICAL-1 (Dec 2025): Signals when exploration should be boosted due to
    training anomalies (loss spikes, divergence, stalls). This closes the feedback
    loop from training issues back to selfplay exploration rates.

    Subscribers (e.g., SelfplayScheduler, TemperatureScheduler) should react by:
    - Increasing MCTS exploration temperature
    - Adding more Dirichlet noise at root
    - Prioritizing diverse game openings

    Args:
        config_key: Configuration key (e.g., "hex8_2p")
        boost_factor: Exploration boost multiplier (1.0 = no boost, 2.0 = double)
        reason: Why exploration is being boosted ("loss_anomaly", "stall", "divergence")
        anomaly_count: Number of consecutive anomalies that triggered this boost
        source: Component emitting the event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.EXPLORATION_BOOST,
        payload={
            "config_key": config_key,
            "boost_factor": boost_factor,
            "reason": reason,
            "anomaly_count": anomaly_count,
        },
        source=source,
    ))


# =============================================================================
# Data Sync Events
# =============================================================================

async def emit_data_sync_started(
    host: str,
    sync_type: str = "incremental",
    source: str = "",
) -> None:
    """Emit a DATA_SYNC_STARTED event.

    Args:
        host: Host being synced
        sync_type: Type of sync (incremental, full)
        source: Component initiating the sync
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DATA_SYNC_STARTED,
        payload={
            "host": host,
            "sync_type": sync_type,
        },
        source=source,
    ))


async def emit_data_sync_completed(
    host: str,
    games_synced: int,
    duration: float,
    bytes_transferred: int = 0,
    source: str = "",
    avg_quality_score: float = 0.0,
    high_quality_count: int = 0,
    config: str = "",
) -> None:
    """Emit a DATA_SYNC_COMPLETED event with quality metrics.

    Args:
        host: Host that was synced
        games_synced: Number of games transferred
        duration: Sync duration in seconds
        bytes_transferred: Bytes transferred (if known)
        source: Component that performed the sync
        avg_quality_score: Average quality score of synced games (0-1)
        high_quality_count: Number of games with quality >= 0.7
        config: Configuration key (e.g., "square8_2p")
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DATA_SYNC_COMPLETED,
        payload={
            "host": host,
            "games_synced": games_synced,
            "duration": duration,
            "bytes_transferred": bytes_transferred,
            "avg_quality_score": avg_quality_score,
            "high_quality_count": high_quality_count,
            "config": config,
        },
        source=source,
    ))


async def emit_data_sync_failed(
    host: str,
    error: str,
    retry_count: int = 0,
    source: str = "",
) -> None:
    """Emit a DATA_SYNC_FAILED event.

    Args:
        host: Host that failed to sync
        error: Error message
        retry_count: Number of retries attempted
        source: Component that attempted the sync
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DATA_SYNC_FAILED,
        payload={
            "host": host,
            "error": error,
            "retry_count": retry_count,
        },
        source=source,
    ))


# =============================================================================
# Model Distribution Events (December 2025)
# =============================================================================

async def emit_model_distribution_started(
    total_models: int,
    target_hosts: int,
    source: str = "",
) -> None:
    """Emit a MODEL_DISTRIBUTION_STARTED event.

    December 2025: Enables tracking of model sync operations.

    Args:
        total_models: Number of models being distributed
        target_hosts: Number of target hosts
        source: Component initiating distribution
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.MODEL_DISTRIBUTION_STARTED,
        payload={
            "total_models": total_models,
            "target_hosts": target_hosts,
        },
        source=source,
    ))


async def emit_model_distribution_complete(
    models_collected: int,
    models_distributed: int,
    target_hosts: int,
    duration: float,
    source: str = "",
) -> None:
    """Emit a MODEL_DISTRIBUTION_COMPLETE event.

    December 2025: Enables tracking of successful model sync completion.

    Args:
        models_collected: Number of models collected from cluster
        models_distributed: Number of models distributed to cluster
        target_hosts: Number of target hosts
        duration: Distribution duration in seconds
        source: Component that completed distribution
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.MODEL_DISTRIBUTION_COMPLETE,
        payload={
            "models_collected": models_collected,
            "models_distributed": models_distributed,
            "target_hosts": target_hosts,
            "duration": duration,
        },
        source=source,
    ))


async def emit_model_distribution_failed(
    error: str,
    partial_models: int = 0,
    source: str = "",
) -> None:
    """Emit a MODEL_DISTRIBUTION_FAILED event.

    December 2025: Enables tracking of failed model sync operations.

    Args:
        error: Error message
        partial_models: Number of models partially synced before failure
        source: Component where distribution failed
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.MODEL_DISTRIBUTION_FAILED,
        payload={
            "error": error,
            "partial_models": partial_models,
        },
        source=source,
    ))


# =============================================================================
# Host Status Events
# =============================================================================

async def emit_host_online(
    host: str,
    capabilities: list[str] | None = None,
    source: str = "",
) -> None:
    """Emit a HOST_ONLINE event.

    Args:
        host: Host that came online
        capabilities: List of host capabilities (gpu, cpu, etc.)
        source: Component that detected the host
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.HOST_ONLINE,
        payload={
            "host": host,
            "capabilities": capabilities or [],
        },
        source=source,
    ))


async def emit_host_offline(
    host: str,
    reason: str = "",
    last_seen: float | None = None,
    source: str = "",
) -> None:
    """Emit a HOST_OFFLINE event.

    Args:
        host: Host that went offline
        reason: Reason for going offline (timeout, error, etc.)
        last_seen: Timestamp when host was last seen
        source: Component that detected the offline status
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.HOST_OFFLINE,
        payload={
            "host": host,
            "reason": reason,
            "last_seen": last_seen,
        },
        source=source,
    ))


# =============================================================================
# Daemon Lifecycle Events
# =============================================================================

async def emit_daemon_started(
    daemon_name: str,
    hostname: str,
    pid: int,
    source: str = "",
) -> None:
    """Emit a DAEMON_STARTED event.

    Args:
        daemon_name: Name of the daemon
        hostname: Host running the daemon
        pid: Process ID
        source: Component starting the daemon
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DAEMON_STARTED,
        payload={
            "daemon_name": daemon_name,
            "hostname": hostname,
            "pid": pid,
        },
        source=source,
    ))


async def emit_daemon_stopped(
    daemon_name: str,
    hostname: str,
    reason: str = "normal",
    source: str = "",
) -> None:
    """Emit a DAEMON_STOPPED event.

    Args:
        daemon_name: Name of the daemon
        hostname: Host running the daemon
        reason: Reason for stopping (normal, error, signal)
        source: Component reporting the stop
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DAEMON_STOPPED,
        payload={
            "daemon_name": daemon_name,
            "hostname": hostname,
            "reason": reason,
        },
        source=source,
    ))


async def emit_daemon_permanently_failed(
    daemon_name: str,
    hostname: str,
    restart_count: int,
    source: str = "",
) -> None:
    """Emit a DAEMON_PERMANENTLY_FAILED event.

    December 2025: Emitted when a daemon has exceeded its hourly restart limit
    and is marked as permanently failed. This requires manual intervention to
    clear the failure state and allow the daemon to restart.

    Args:
        daemon_name: Name of the daemon
        hostname: Host running the daemon
        restart_count: Number of restarts in the last hour
        source: Component reporting the failure
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DAEMON_PERMANENTLY_FAILED,
        payload={
            "daemon_name": daemon_name,
            "hostname": hostname,
            "restart_count": restart_count,
            "requires_intervention": True,
        },
        source=source,
    ))


async def emit_daemon_crash_loop_detected(
    daemon_name: str,
    hostname: str,
    restart_count: int,
    window_minutes: int,
    max_restarts: int,
    source: str = "",
) -> None:
    """Emit a DAEMON_CRASH_LOOP_DETECTED event (early warning before permanent failure).

    December 2025: Emitted when a daemon has restarted multiple times within a short
    window, indicating a crash loop. This is an early warning before the daemon
    reaches the permanent failure threshold (MAX_RESTARTS_PER_HOUR).

    This event enables:
    - Dashboard alerts for crash loops
    - Proactive investigation before permanent failure
    - Notification to operators

    Args:
        daemon_name: Name of the daemon
        hostname: Host running the daemon
        restart_count: Number of restarts in the window
        window_minutes: Time window in minutes
        max_restarts: Max restarts before permanent failure
        source: Component reporting the event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DAEMON_CRASH_LOOP_DETECTED,
        payload={
            "daemon_name": daemon_name,
            "hostname": hostname,
            "restart_count": restart_count,
            "window_minutes": window_minutes,
            "max_restarts": max_restarts,
            "severity": "warning",
            "message": f"{daemon_name} is crash looping ({restart_count} restarts in {window_minutes}min)",
        },
        source=source,
    ))


# =============================================================================
# Orphan Detection Events (December 2025)
# =============================================================================

async def emit_orphan_games_detected(
    host: str,
    orphan_count: int,
    orphan_paths: list[str],
    total_games: int = 0,
    source: str = "",
) -> None:
    """Emit an ORPHAN_GAMES_DETECTED event.

    Dec 2025: Emitted when unregistered game databases are found on a node.
    These are databases that exist on disk but aren't tracked in the manifest.

    Args:
        host: Host where orphans were detected
        orphan_count: Number of orphan databases found
        orphan_paths: List of paths to orphan databases
        total_games: Total game count in orphan databases
        source: Component that detected the orphans
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.ORPHAN_GAMES_DETECTED,
        payload={
            "host": host,
            "orphan_count": orphan_count,
            "orphan_paths": orphan_paths[:10],  # Limit to 10 paths to avoid huge payloads
            "total_games": total_games,
        },
        source=source,
    ))


async def emit_orphan_games_registered(
    host: str,
    registered_count: int,
    registered_paths: list[str],
    games_recovered: int = 0,
    source: str = "",
) -> None:
    """Emit an ORPHAN_GAMES_REGISTERED event.

    Dec 2025: Emitted when orphan databases have been auto-registered
    into the manifest after detection.

    Args:
        host: Host where orphans were registered
        registered_count: Number of databases registered
        registered_paths: List of paths that were registered
        games_recovered: Total game count recovered from orphans
        source: Component that performed registration
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.ORPHAN_GAMES_REGISTERED,
        payload={
            "host": host,
            "registered_count": registered_count,
            "registered_paths": registered_paths[:10],  # Limit paths
            "games_recovered": games_recovered,
        },
        source=source,
    ))


# =============================================================================
# Training Failure Events
# =============================================================================

async def emit_training_started(
    config: str,
    model_path: str | None = None,
    source: str = "",
) -> None:
    """Emit a TRAINING_STARTED event.

    Args:
        config: Board configuration
        model_path: Path to model being trained
        source: Component starting the training
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_STARTED,
        payload={
            "config": config,
            "model_path": model_path,
        },
        source=source,
    ))


async def emit_training_failed(
    config: str,
    error: str,
    duration: float = 0,
    source: str = "",
) -> None:
    """Emit a TRAINING_FAILED event.

    Args:
        config: Board configuration
        error: Error message
        duration: How long training ran before failing
        source: Component reporting the failure
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_FAILED,
        payload={
            "config": config,
            "error": error,
            "duration": duration,
        },
        source=source,
    ))


# =============================================================================
# Evaluation Failure Events
# =============================================================================

async def emit_evaluation_started(
    config: str,
    model_id: str,
    games_planned: int = 0,
    source: str = "",
) -> None:
    """Emit an EVALUATION_STARTED event.

    Args:
        config: Board configuration
        model_id: Model being evaluated
        games_planned: Number of evaluation games planned
        source: Component starting the evaluation
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.EVALUATION_STARTED,
        payload={
            "config": config,
            "model_id": model_id,
            "games_planned": games_planned,
        },
        source=source,
    ))


async def emit_evaluation_failed(
    config: str,
    model_id: str,
    error: str,
    source: str = "",
) -> None:
    """Emit an EVALUATION_FAILED event.

    Args:
        config: Board configuration
        model_id: Model being evaluated
        error: Error message
        source: Component reporting the failure
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.EVALUATION_FAILED,
        payload={
            "config": config,
            "model_id": model_id,
            "error": error,
        },
        source=source,
    ))


# =============================================================================
# Promotion Events
# =============================================================================

async def emit_promotion_candidate(
    model_id: str,
    board_type: str,
    num_players: int,
    win_rate_vs_heuristic: float,
    source: str = "",
) -> None:
    """Emit a PROMOTION_CANDIDATE event.

    Args:
        model_id: Model that is a promotion candidate
        board_type: Board type (e.g., "hex8", "square8")
        num_players: Number of players (2, 3, or 4)
        win_rate_vs_heuristic: Win rate against heuristic baseline
        source: Component emitting the event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.PROMOTION_CANDIDATE,
        payload={
            "model_id": model_id,
            "board_type": board_type,
            "num_players": num_players,
            "config_key": f"{board_type}_{num_players}p",
            "win_rate_vs_heuristic": win_rate_vs_heuristic,
        },
        source=source,
    ))


async def emit_promotion_started(
    config: str,
    model_id: str,
    source: str = "",
) -> None:
    """Emit a PROMOTION_STARTED event.

    Args:
        config: Board configuration
        model_id: Model being considered for promotion
        source: Component starting the promotion
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.PROMOTION_STARTED,
        payload={
            "config": config,
            "model_id": model_id,
        },
        source=source,
    ))


async def emit_promotion_failed(
    config: str,
    model_id: str,
    error: str,
    source: str = "",
) -> None:
    """Emit a PROMOTION_FAILED event.

    Args:
        config: Board configuration
        model_id: Model that failed promotion
        error: Error message
        source: Component reporting the failure
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.PROMOTION_FAILED,
        payload={
            "config": config,
            "model_id": model_id,
            "error": error,
        },
        source=source,
    ))


async def emit_promotion_rejected(
    config: str,
    model_id: str,
    reason: str,
    elo_improvement: float = 0,
    source: str = "",
) -> None:
    """Emit a PROMOTION_REJECTED event.

    Args:
        config: Board configuration
        model_id: Model that was rejected
        reason: Reason for rejection
        elo_improvement: Elo improvement achieved (if any)
        source: Component rejecting the promotion
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.PROMOTION_REJECTED,
        payload={
            "config": config,
            "model_id": model_id,
            "reason": reason,
            "elo_improvement": elo_improvement,
        },
        source=source,
    ))


# =============================================================================
# Tier Promotion Events
# =============================================================================
# STATUS: RESERVED - Emitter exists, no subscribers yet (Dec 2025)
#
# Purpose: Track model promotion through difficulty ladder tiers (D1→D2→D3, etc.)
#
# Intended subscribers (to be implemented Q1 2026):
#   - CurriculumIntegration: Adjust training curriculum when tier changes
#   - TierProgressTracker: Monitor promotion velocity for dashboards
#   - ModelRegistry: Update model metadata with tier information
#
# The emit function is wired from app.training.promotion_controller._execute_tier_promotion()
# when a model passes threshold evaluation at its current tier.
# =============================================================================

async def emit_tier_promotion(
    config: str,
    old_tier: str,
    new_tier: str,
    model_id: str = "",
    win_rate: float = 0.0,
    elo: float = 0.0,
    games_played: int = 0,
    source: str = "",
) -> None:
    """Emit a TIER_PROMOTION event for difficulty ladder progression.

    Args:
        config: Board configuration (e.g., "square8_2p")
        old_tier: Previous tier (e.g., "D4")
        new_tier: New tier after promotion (e.g., "D5")
        model_id: ID of the model being promoted
        win_rate: Win rate that triggered promotion
        elo: Current Elo rating
        games_played: Number of games played at current tier
        source: Component that triggered the promotion
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TIER_PROMOTION,
        payload={
            "config": config,
            "old_tier": old_tier,
            "new_tier": new_tier,
            "model_id": model_id,
            "win_rate": win_rate,
            "elo": elo,
            "games_played": games_played,
        },
        source=source,
    ))


async def emit_deadlock_detected(
    resources: list[str],
    holders: list[str],
    source: str = "",
) -> None:
    """Emit a DEADLOCK_DETECTED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DEADLOCK_DETECTED,
        payload={
            "resources": resources,
            "holders": holders,
        },
        source=source,
    ))


# =============================================================================
# Checkpoint Events (December 2025)
# =============================================================================

async def emit_checkpoint_saved(
    config: str,
    checkpoint_path: str,
    epoch: int = 0,
    step: int = 0,
    metrics: dict[str, float] | None = None,
    source: str = "",
) -> None:
    """Emit a CHECKPOINT_SAVED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CHECKPOINT_SAVED,
        payload={
            "config": config,
            "checkpoint_path": checkpoint_path,
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
        },
        source=source,
    ))


async def emit_checkpoint_loaded(
    config: str,
    checkpoint_path: str,
    epoch: int = 0,
    step: int = 0,
    source: str = "",
) -> None:
    """Emit a CHECKPOINT_LOADED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CHECKPOINT_LOADED,
        payload={
            "config": config,
            "checkpoint_path": checkpoint_path,
            "epoch": epoch,
            "step": step,
        },
        source=source,
    ))


# =============================================================================
# Task Lifecycle Events (December 2025)
# =============================================================================

async def emit_task_spawned(
    task_id: str,
    task_type: str,
    node_id: str,
    config: str = "",
    priority: int = 0,
    source: str = "",
) -> None:
    """Emit a TASK_SPAWNED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TASK_SPAWNED,
        payload={
            "task_id": task_id,
            "task_type": task_type,
            "node_id": node_id,
            "config": config,
            "priority": priority,
        },
        source=source,
    ))


async def emit_task_heartbeat(
    task_id: str,
    node_id: str,
    progress: float = 0.0,
    status: str = "running",
    source: str = "",
) -> None:
    """Emit a TASK_HEARTBEAT event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TASK_HEARTBEAT,
        payload={
            "task_id": task_id,
            "node_id": node_id,
            "progress": progress,
            "status": status,
        },
        source=source,
    ))


async def emit_task_completed(
    task_id: str,
    task_type: str,
    node_id: str,
    duration_seconds: float = 0.0,
    result: dict[str, Any] | None = None,
    source: str = "",
) -> None:
    """Emit a TASK_COMPLETED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TASK_COMPLETED,
        payload={
            "task_id": task_id,
            "task_type": task_type,
            "node_id": node_id,
            "duration_seconds": duration_seconds,
            "result": result or {},
        },
        source=source,
    ))


async def emit_task_failed(
    task_id: str,
    task_type: str,
    node_id: str,
    error: str,
    duration_seconds: float = 0.0,
    source: str = "",
) -> None:
    """Emit a TASK_FAILED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TASK_FAILED,
        payload={
            "task_id": task_id,
            "task_type": task_type,
            "node_id": node_id,
            "error": error,
            "duration_seconds": duration_seconds,
        },
        source=source,
    ))


async def emit_task_orphaned(
    task_id: str,
    task_type: str,
    node_id: str,
    last_heartbeat_seconds_ago: float = 0.0,
    source: str = "",
) -> None:
    """Emit a TASK_ORPHANED event for tasks that stopped sending heartbeats."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TASK_ORPHANED,
        payload={
            "task_id": task_id,
            "task_type": task_type,
            "node_id": node_id,
            "last_heartbeat_seconds_ago": last_heartbeat_seconds_ago,
        },
        source=source,
    ))


async def emit_task_abandoned(
    work_id: str,
    reason: str,
    component: str = "p2p_orchestrator",
    node_id: str = "",
    timestamp: float | None = None,
    source: str = "",
) -> None:
    """Emit a TASK_ABANDONED event for intentionally cancelled tasks.

    December 2025: Added to distinguish intentional cancellations from failures.
    Use this when a task is deliberately stopped (e.g., user request, superseded
    by newer version, resource reallocation) rather than failing unexpectedly.

    Args:
        work_id: The work item ID that was abandoned
        reason: Why the task was abandoned (e.g., "superseded", "user_request")
        component: Which component triggered the abandonment
        node_id: Optional node where the task was running
        timestamp: When abandonment occurred (defaults to now)
        source: Event source identifier
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TASK_ABANDONED,
        payload={
            "work_id": work_id,
            "reason": reason,
            "component": component,
            "node_id": node_id,
            "timestamp": timestamp or time.time(),
        },
        source=source,
    ))


# =============================================================================
# Job Spawn Verification Events (January 2026 - Sprint 6)
# =============================================================================


async def emit_job_spawn_verified(
    job_id: str,
    node_id: str,
    config_key: str,
    verification_time_seconds: float,
    source: str = "",
) -> None:
    """Emit a JOB_SPAWN_VERIFIED event when a job is confirmed running.

    January 2026 - Sprint 6: Added for job spawn verification loop.
    Used by SelfplayScheduler to track spawn success rates and adjust capacity estimates.

    Args:
        job_id: The job ID that was verified
        node_id: Node where the job is running
        config_key: Board configuration key (e.g., "hex8_2p")
        verification_time_seconds: Time taken to verify the spawn
        source: Event source identifier
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.JOB_SPAWN_VERIFIED,
        payload={
            "job_id": job_id,
            "node_id": node_id,
            "config_key": config_key,
            "verification_time_seconds": verification_time_seconds,
            "timestamp": time.time(),
        },
        source=source,
    ))


async def emit_job_spawn_failed(
    job_id: str,
    node_id: str,
    config_key: str,
    timeout_seconds: float,
    reason: str = "verification_timeout",
    source: str = "",
) -> None:
    """Emit a JOB_SPAWN_FAILED event when job spawn verification fails.

    January 2026 - Sprint 6: Added for job spawn verification loop.
    Used by SelfplayScheduler to detect spawn failures and adjust node capacity.

    Args:
        job_id: The job ID that failed to spawn
        node_id: Node where spawn was attempted
        config_key: Board configuration key (e.g., "hex8_2p")
        timeout_seconds: Verification timeout that was exceeded
        reason: Why verification failed (e.g., "verification_timeout", "job_not_found")
        source: Event source identifier
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.JOB_SPAWN_FAILED,
        payload={
            "job_id": job_id,
            "node_id": node_id,
            "config_key": config_key,
            "timeout_seconds": timeout_seconds,
            "reason": reason,
            "timestamp": time.time(),
        },
        source=source,
    ))


# =============================================================================
# Capacity/Backpressure Events (December 2025)
# =============================================================================

async def emit_cluster_capacity_changed(
    total_gpus: int,
    available_gpus: int,
    total_nodes: int,
    healthy_nodes: int,
    source: str = "",
) -> None:
    """Emit a CLUSTER_CAPACITY_CHANGED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CLUSTER_CAPACITY_CHANGED,
        payload={
            "total_gpus": total_gpus,
            "available_gpus": available_gpus,
            "total_nodes": total_nodes,
            "healthy_nodes": healthy_nodes,
        },
        source=source,
    ))


async def emit_backpressure_activated(
    reason: str,
    queue_depth: int = 0,
    utilization_percent: float = 0.0,
    source: str = "",
) -> None:
    """Emit a BACKPRESSURE_ACTIVATED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.BACKPRESSURE_ACTIVATED,
        payload={
            "reason": reason,
            "queue_depth": queue_depth,
            "utilization_percent": utilization_percent,
        },
        source=source,
    ))


async def emit_backpressure_released(
    reason: str,
    duration_seconds: float = 0.0,
    source: str = "",
) -> None:
    """Emit a BACKPRESSURE_RELEASED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.BACKPRESSURE_RELEASED,
        payload={
            "reason": reason,
            "duration_seconds": duration_seconds,
        },
        source=source,
    ))


# =============================================================================
# Leader Election Events (December 2025)
# =============================================================================

async def emit_leader_elected(
    leader_id: str,
    term: int = 0,
    source: str = "",
) -> None:
    """Emit a LEADER_ELECTED event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.LEADER_ELECTED,
        payload={
            "leader_id": leader_id,
            "term": term,
        },
        source=source,
    ))


async def emit_leader_lost(
    old_leader_id: str,
    reason: str = "",
    source: str = "",
) -> None:
    """Emit a LEADER_LOST event."""
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.LEADER_LOST,
        payload={
            "old_leader_id": old_leader_id,
            "reason": reason,
        },
        source=source,
    ))


async def emit_leader_heartbeat_missing(
    leader_id: str,
    last_heartbeat: float,
    expected_interval: float,
    delay_seconds: float,
    source: str = "",
) -> None:
    """Emit a LEADER_HEARTBEAT_MISSING event for monitoring.

    This event is emitted when the leader hasn't sent a heartbeat for longer
    than expected. It's a warning signal before the lease actually expires.

    Args:
        leader_id: ID of the leader node
        last_heartbeat: Timestamp of last known heartbeat
        expected_interval: Expected heartbeat interval in seconds
        delay_seconds: How many seconds the heartbeat is delayed
        source: Node emitting the event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.LEADER_HEARTBEAT_MISSING,
        payload={
            "leader_id": leader_id,
            "last_heartbeat": last_heartbeat,
            "expected_interval": expected_interval,
            "delay_seconds": delay_seconds,
        },
        source=source,
    ))


async def emit_leader_lease_expired(
    leader_id: str,
    lease_expiry_time: float,
    current_time: float,
    grace_seconds: float = 30.0,
    source: str = "",
) -> None:
    """Emit a LEADER_LEASE_EXPIRED event when leader lease expires without stepdown.

    This event is emitted when a leader's lease expires but the leader hasn't
    voluntarily stepped down. This indicates a potential stuck or unresponsive
    leader that needs to be replaced via election.

    January 2, 2026: Added for stale leader alerting in Sprint 3.

    Args:
        leader_id: ID of the leader node whose lease expired
        lease_expiry_time: Unix timestamp when lease expired
        current_time: Current Unix timestamp
        grace_seconds: Grace period that was allowed beyond expiry
        source: Node emitting the event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.LEADER_LEASE_EXPIRED,
        payload={
            "leader_id": leader_id,
            "lease_expiry_time": lease_expiry_time,
            "current_time": current_time,
            "grace_seconds": grace_seconds,
            "expired_by_seconds": current_time - lease_expiry_time,
        },
        source=source,
    ))


async def emit_training_loss_anomaly(
    config_key: str,
    current_loss: float,
    avg_loss: float,
    epoch: int,
    anomaly_ratio: float = 0.0,
    source: str = "training",
) -> None:
    """Emit a TRAINING_LOSS_ANOMALY event when loss spikes above threshold.

    This triggers quality checks and potential data investigation.
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_LOSS_ANOMALY,
        payload={
            "config_key": config_key,
            "current_loss": current_loss,
            "avg_loss": avg_loss,
            "epoch": epoch,
            "anomaly_ratio": anomaly_ratio,
        },
        source=source,
    ))


async def emit_training_loss_trend(
    config_key: str,
    trend: str,  # "improving", "stalled", "degrading"
    epoch: int,
    current_loss: float,
    previous_loss: float,
    improvement_rate: float = 0.0,
    source: str = "training",
) -> None:
    """Emit a TRAINING_LOSS_TREND event to guide data collection.

    Used by selfplay to adjust exploration/exploitation based on training needs.
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_LOSS_TREND,
        payload={
            "config_key": config_key,
            "trend": trend,
            "epoch": epoch,
            "current_loss": current_loss,
            "previous_loss": previous_loss,
            "improvement_rate": improvement_rate,
        },
        source=source,
    ))


async def emit_training_early_stopped(
    config_key: str,
    epoch: int,
    best_loss: float,
    final_loss: float,
    best_elo: float | None = None,
    reason: str = "loss_stagnation",
    epochs_without_improvement: int = 0,
    source: str = "training",
) -> None:
    """Emit a TRAINING_EARLY_STOPPED event when training stops early.

    This triggers curriculum boost for the config to accelerate training recovery.
    The curriculum feedback system should increase training allocation for
    configs that are stuck.

    Args:
        config_key: Board/player config (e.g., "hex8_2p")
        epoch: Epoch at which training stopped
        best_loss: Best validation loss achieved
        final_loss: Final validation loss before stopping
        best_elo: Best Elo achieved (if Elo tracking enabled)
        reason: Why early stopping triggered ("loss_stagnation", "elo_stagnation", "regression")
        epochs_without_improvement: How many epochs passed without improvement
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.TRAINING_EARLY_STOPPED,
        payload={
            "config_key": config_key,
            "epoch": epoch,
            "best_loss": best_loss,
            "final_loss": final_loss,
            "best_elo": best_elo,
            "reason": reason,
            "epochs_without_improvement": epochs_without_improvement,
        },
        source=source,
    ))


async def emit_sync_stalled(
    source_host: str,
    target_host: str,
    data_type: str,
    timeout_seconds: float,
    retry_count: int = 0,
    source: str = "sync_coordinator",
) -> None:
    """Emit a SYNC_STALLED event when sync times out.

    This triggers failover to alternative sync sources.
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.SYNC_STALLED,
        payload={
            "source_host": source_host,
            "target_host": target_host,
            "data_type": data_type,
            "timeout_seconds": timeout_seconds,
            "retry_count": retry_count,
        },
        source=source,
    ))


async def emit_node_overloaded(
    host: str,
    cpu_percent: float,
    gpu_percent: float = 0.0,
    memory_percent: float = 0.0,
    resource_type: str = "cpu",
    source: str = "health_manager",
) -> None:
    """Emit a NODE_OVERLOADED event for job redistribution.

    This triggers the job scheduler to migrate jobs away from overloaded nodes.
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NODE_OVERLOADED,
        payload={
            "host": host,
            "cpu_percent": cpu_percent,
            "gpu_percent": gpu_percent,
            "memory_percent": memory_percent,
            "resource_type": resource_type,
        },
        source=source,
    ))


async def emit_selfplay_target_updated(
    config_key: str,
    target_games: int,
    reason: str,
    priority: int = 5,
    source: str = "feedback_controller",
) -> None:
    """Emit a SELFPLAY_TARGET_UPDATED event to scale selfplay.

    Used to request more selfplay games when training needs data.
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.SELFPLAY_TARGET_UPDATED,
        payload={
            "config_key": config_key,
            "target_games": target_games,
            "reason": reason,
            "priority": priority,
        },
        source=source,
    ))


async def emit_selfplay_complete(
    config_key: str,
    games_played: int,
    db_path: str = "",
    duration_seconds: float = 0.0,
    engine: str = "unknown",
    source: str = "selfplay_runner",
) -> None:
    """Emit a SELFPLAY_COMPLETE event when a selfplay batch finishes.

    P0.2 Dec 2025: Bridges StageEvent.SELFPLAY_COMPLETE to DataEventBus.
    This enables components subscribed to DataEventType.SELFPLAY_COMPLETE
    (e.g., feedback_loop_controller, master_loop) to receive the event.

    Args:
        config_key: Board config (e.g., "hex8_2p")
        games_played: Number of games completed
        db_path: Path to database with games
        duration_seconds: Total selfplay duration
        engine: Selfplay engine used (e.g., "gumbel", "heuristic")
        source: Component that ran selfplay
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.SELFPLAY_COMPLETE,
        payload={
            "config_key": config_key,
            "games_played": games_played,
            "db_path": db_path,
            "duration_seconds": duration_seconds,
            "engine": engine,
        },
        source=source,
    ))


async def emit_idle_resource_detected(
    node_id: str,
    host: str,
    gpu_utilization: float,
    gpu_memory_gb: float,
    idle_duration_seconds: float,
    recommended_config: str = "",
    source: str = "idle_resource_daemon",
) -> None:
    """Emit an IDLE_RESOURCE_DETECTED event for job scheduling.

    Used by IdleResourceDaemon to signal idle GPU resources that can be used.
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.IDLE_RESOURCE_DETECTED,
        payload={
            "node_id": node_id,
            "host": host,
            "gpu_utilization": gpu_utilization,
            "gpu_memory_gb": gpu_memory_gb,
            "idle_duration_seconds": idle_duration_seconds,
            "recommended_config": recommended_config,
        },
        source=source,
    ))


# =============================================================================
# Batch Scheduling Events (December 27, 2025)
# =============================================================================
# P0 fix for pipeline coordination - emits when batches are scheduled/dispatched


async def emit_batch_scheduled(
    batch_id: str,
    batch_type: str,
    config_key: str,
    job_count: int,
    target_nodes: list[str],
    reason: str = "normal",
    source: str = "selfplay_scheduler",
) -> None:
    """Emit a BATCH_SCHEDULED event when a batch of jobs is scheduled.

    Used by SelfplayScheduler to signal when a batch of jobs has been selected
    for dispatch. Enables pipeline coordination to track batch operations.

    Args:
        batch_id: Unique batch identifier
        batch_type: Type of batch ("selfplay", "training", "tournament")
        config_key: Configuration key (e.g., "hex8_2p")
        job_count: Number of jobs in the batch
        target_nodes: List of node IDs selected for dispatch
        reason: Why the batch was scheduled (e.g., "priority_change", "idle_detection")
        source: Component that scheduled the batch
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.BATCH_SCHEDULED,
        payload={
            "batch_id": batch_id,
            "batch_type": batch_type,
            "config_key": config_key,
            "job_count": job_count,
            "target_nodes": target_nodes,
            "reason": reason,
            "timestamp": time.time(),
        },
        source=source,
    ))


async def emit_batch_dispatched(
    batch_id: str,
    batch_type: str,
    config_key: str,
    jobs_dispatched: int,
    jobs_failed: int = 0,
    target_nodes: list[str] | None = None,
    source: str = "job_dispatcher",
) -> None:
    """Emit a BATCH_DISPATCHED event when batch jobs are sent to nodes.

    Used after jobs have been dispatched to signal the pipeline that work
    is in progress.

    Args:
        batch_id: Unique batch identifier (same as BATCH_SCHEDULED)
        batch_type: Type of batch ("selfplay", "training", "tournament")
        config_key: Configuration key (e.g., "hex8_2p")
        jobs_dispatched: Number of jobs successfully dispatched
        jobs_failed: Number of jobs that failed to dispatch
        target_nodes: List of node IDs that received jobs
        source: Component that dispatched the batch
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.BATCH_DISPATCHED,
        payload={
            "batch_id": batch_id,
            "batch_type": batch_type,
            "config_key": config_key,
            "jobs_dispatched": jobs_dispatched,
            "jobs_failed": jobs_failed,
            "target_nodes": target_nodes or [],
            "timestamp": time.time(),
        },
        source=source,
    ))


# =============================================================================
# P0.5 Missing Event Emitters (December 2025)
# =============================================================================
# These were identified as "ghost events" - event types defined but never emitted.


async def emit_data_stale(
    config: str,
    data_age_hours: float,
    last_sync_time: str,
    threshold_hours: float = 1.0,
    source: str = "training_freshness",
) -> None:
    """Emit a DATA_STALE event when training data is older than threshold.

    P0.5 Dec 2025: Closes feedback loop for data freshness monitoring.
    This enables training_freshness.py and sync daemons to respond to stale data.

    Args:
        config: Board configuration (e.g., "hex8_2p")
        data_age_hours: How old the data is in hours
        last_sync_time: ISO timestamp of last data sync
        threshold_hours: Threshold at which data is considered stale
        source: Component that detected staleness
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DATA_STALE,
        payload={
            "config": config,
            "data_age_hours": data_age_hours,
            "last_sync_time": last_sync_time,
            "threshold_hours": threshold_hours,
        },
        source=source,
    ))


async def emit_curriculum_advanced(
    config: str,
    old_tier: str,
    new_tier: str,
    trigger_reason: str,
    elo: float = 0.0,
    win_rate: float = 0.0,
    games_at_tier: int = 0,
    source: str = "curriculum_integration",
) -> None:
    """Emit a CURRICULUM_ADVANCED event when curriculum tier progresses.

    P0.5 Dec 2025: Closes curriculum feedback loop.
    This enables SelfplayScheduler to adjust priorities based on curriculum progression.

    Args:
        config: Board configuration (e.g., "hex8_2p")
        old_tier: Previous curriculum tier (e.g., "BEGINNER")
        new_tier: New curriculum tier (e.g., "INTERMEDIATE")
        trigger_reason: What triggered the advancement (elo, winrate, games)
        elo: Current Elo rating
        win_rate: Current win rate
        games_at_tier: Number of games played at old tier
        source: Component that triggered advancement
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CURRICULUM_ADVANCED,
        payload={
            "config": config,
            "old_tier": old_tier,
            "new_tier": new_tier,
            "trigger_reason": trigger_reason,
            "elo": elo,
            "win_rate": win_rate,
            "games_at_tier": games_at_tier,
        },
        source=source,
    ))


async def emit_curriculum_rollback_completed(
    config_key: str,
    old_weight: float,
    new_weight: float,
    elo_delta: float,
    trigger_reason: str = "regression_detected",
    source: str = "curriculum_integration",
) -> None:
    """Emit a CURRICULUM_ROLLBACK_COMPLETED event when curriculum weight is reduced.

    Sprint 16.1 (Jan 3, 2026): Confirmation event for observability when curriculum
    weight is rolled back due to regression or other quality issues. Enables:
    - Monitoring dashboards to track rollback frequency
    - SelfplayScheduler to adjust priorities after rollback
    - Alert systems to notify on repeated rollbacks

    Args:
        config_key: Board configuration (e.g., "hex8_2p")
        old_weight: Previous curriculum weight (0.0-1.0)
        new_weight: New reduced weight (0.0-1.0)
        elo_delta: Elo change that triggered rollback (negative for regression)
        trigger_reason: What triggered the rollback (e.g., "regression_detected")
        source: Component that triggered rollback
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CURRICULUM_ROLLBACK_COMPLETED,
        payload={
            "config_key": config_key,
            "old_weight": old_weight,
            "new_weight": new_weight,
            "elo_delta": elo_delta,
            "trigger_reason": trigger_reason,
            "weight_reduction_pct": (1 - new_weight / old_weight) * 100 if old_weight > 0 else 0,
        },
        source=source,
    ))


async def emit_daemon_status_changed(
    daemon_name: str,
    hostname: str,
    old_status: str,
    new_status: str,
    reason: str = "",
    error: str | None = None,
    source: str = "daemon_manager",
) -> None:
    """Emit a DAEMON_STATUS_CHANGED event for daemon health transitions.

    P0.5 Dec 2025: Enables watchdog to react to daemon health state changes.
    Useful for detecting stuck, crashed, or restarted daemons.

    Args:
        daemon_name: Name of the daemon (e.g., "selfplay_scheduler")
        hostname: Host running the daemon
        old_status: Previous status (running, stopped, error)
        new_status: New status (running, stopped, error, stuck)
        reason: Why the status changed (timeout, exception, signal, restart)
        error: Error message if transition was due to error
        source: Component reporting the change
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DAEMON_STATUS_CHANGED,
        payload={
            "daemon_name": daemon_name,
            "hostname": hostname,
            "old_status": old_status,
            "new_status": new_status,
            "reason": reason,
            "error": error,
        },
        source=source,
    ))


async def emit_data_fresh(
    config: str,
    data_age_hours: float,
    last_sync_time: str,
    threshold_hours: float = 1.0,
    source: str = "training_freshness",
) -> None:
    """Emit a DATA_FRESH event when training data meets freshness requirements.

    P0.5 Dec 2025: Closes feedback loop for data freshness monitoring.
    Counterpart to DATA_STALE - signals that sync is NOT needed.

    Args:
        config: Board configuration (e.g., "hex8_2p")
        data_age_hours: How old the data is in hours
        last_sync_time: ISO timestamp of last data sync
        threshold_hours: Threshold at which data is considered stale
        source: Component that checked freshness
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DATA_FRESH,
        payload={
            "config": config,
            "data_age_hours": data_age_hours,
            "last_sync_time": last_sync_time,
            "threshold_hours": threshold_hours,
        },
        source=source,
    ))


async def emit_sync_triggered(
    config: str,
    reason: str,
    source_host: str = "",
    target_hosts: list[str] | None = None,
    data_age_hours: float = 0.0,
    source: str = "auto_sync_daemon",
) -> None:
    """Emit a SYNC_TRIGGERED event when data sync is initiated due to staleness.

    P0.5 Dec 2025: Closes feedback loop for sync coordination.
    Enables components to track when syncs are triggered and why.

    Args:
        config: Board configuration (e.g., "hex8_2p")
        reason: Why sync was triggered (stale_data, manual_request, new_games)
        source_host: Host that initiated the sync
        target_hosts: List of hosts being synced to
        data_age_hours: How stale the data was when sync was triggered
        source: Component that triggered the sync
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.SYNC_TRIGGERED,
        payload={
            "config": config,
            "reason": reason,
            "source_host": source_host,
            "target_hosts": target_hosts or [],
            "data_age_hours": data_age_hours,
        },
        source=source,
    ))


# =============================================================================
# Curriculum Weight Events
# =============================================================================
# STATUS: RESERVED - Emitter exists, no subscribers yet (Dec 2025)
#
# Purpose: Track changes to curriculum/opponent weights for auditing and debugging.
# When curriculum integration adjusts weights (exploration, opponent strength, etc.),
# this event provides a trail for understanding why weights changed.
#
# Intended subscribers (to be implemented Q1 2026):
#   - WeightHistoryTracker: Store weight change history for analysis
#   - CurriculumDashboard: Visualize weight evolution over time
#   - SelfplayScheduler: Could adjust priorities based on weight changes
#
# The emit function is intended to be called from curriculum integration components
# when they adjust training weights.
# =============================================================================

async def emit_weight_updated(
    config: str,
    component: str,
    old_weight: float,
    new_weight: float,
    reason: str,
    trigger_event: str = "",
    source: str = "curriculum_integration",
) -> None:
    """Emit a WEIGHT_UPDATED event when curriculum/opponent weights change.

    P0.5 Dec 2025: Closes curriculum feedback loop.
    Enables tracking of weight changes for debugging and auditing.

    Args:
        config: Board configuration (e.g., "hex8_2p")
        component: What weight changed (opponent, curriculum_tier, exploration)
        old_weight: Previous weight value
        new_weight: New weight value
        reason: Why the weight changed (elo_velocity, win_rate, manual)
        trigger_event: Event that triggered this change (if any)
        source: Component that changed the weight
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.WEIGHT_UPDATED,
        payload={
            "config": config,
            "component": component,
            "old_weight": old_weight,
            "new_weight": new_weight,
            "reason": reason,
            "trigger_event": trigger_event,
        },
        source=source,
    ))


# =============================================================================
# Cluster Health Event Emitters (December 2025 - Phase 21)
# =============================================================================
# These enable key daemons to emit cluster health events for visibility.


async def emit_node_unhealthy(
    node_id: str,
    reason: str,
    *,
    node_ip: str = "",
    gpu_utilization: float | None = None,
    disk_used_percent: float | None = None,
    consecutive_failures: int = 0,
    source: str = "",
) -> None:
    """Emit NODE_UNHEALTHY event when a node is detected as unhealthy.

    Args:
        node_id: Identifier for the node
        reason: Why the node is unhealthy
        node_ip: Node IP address
        gpu_utilization: GPU utilization percentage (0-100)
        disk_used_percent: Disk usage percentage (0-100)
        consecutive_failures: Number of consecutive health check failures
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NODE_UNHEALTHY,
        payload={
            "node_id": node_id,
            "reason": reason,
            "node_ip": node_ip,
            "gpu_utilization": gpu_utilization,
            "disk_used_percent": disk_used_percent,
            "consecutive_failures": consecutive_failures,
        },
        source=source,
    ))


async def emit_node_recovered(
    node_id: str,
    *,
    node_ip: str = "",
    recovery_time_seconds: float = 0.0,
    source: str = "",
) -> None:
    """Emit NODE_RECOVERED event when a node recovers to healthy state.

    Args:
        node_id: Identifier for the node
        node_ip: Node IP address
        recovery_time_seconds: How long recovery took
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NODE_RECOVERED,
        payload={
            "node_id": node_id,
            "node_ip": node_ip,
            "recovery_time_seconds": recovery_time_seconds,
        },
        source=source,
    ))


async def emit_node_suspect(
    node_id: str,
    *,
    node_ip: str = "",
    last_seen: float | None = None,
    seconds_since_heartbeat: float = 0.0,
    source: str = "",
) -> None:
    """Emit NODE_SUSPECT event when a node enters SUSPECT state.

    December 2025: Added for peer state transition tracking.
    SUSPECT is a grace period between ALIVE and DEAD to reduce false positives.

    Args:
        node_id: Identifier for the node
        node_ip: Node IP address
        last_seen: Timestamp when node was last seen
        seconds_since_heartbeat: Seconds since last heartbeat
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NODE_SUSPECT,
        payload={
            "node_id": node_id,
            "node_ip": node_ip,
            "last_seen": last_seen,
            "seconds_since_heartbeat": seconds_since_heartbeat,
        },
        source=source,
    ))


async def emit_node_retired(
    node_id: str,
    *,
    node_ip: str = "",
    reason: str = "manual",
    last_seen: float | None = None,
    total_uptime_seconds: float = 0.0,
    source: str = "",
) -> None:
    """Emit NODE_RETIRED event when a node is retired from the cluster.

    December 2025: Added for peer state transition tracking.
    Retired nodes are excluded from job allocation but may be recovered later.

    Args:
        node_id: Identifier for the node
        node_ip: Node IP address
        reason: Why the node was retired (manual, timeout, error, capacity)
        last_seen: Timestamp when node was last seen
        total_uptime_seconds: Total uptime before retirement
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NODE_RETIRED,
        payload={
            "node_id": node_id,
            "node_ip": node_ip,
            "reason": reason,
            "last_seen": last_seen,
            "total_uptime_seconds": total_uptime_seconds,
        },
        source=source,
    ))


async def emit_node_incompatible_with_workload(
    node_id: str,
    *,
    node_ip: str = "",
    gpu_vram_gb: float = 0.0,
    has_gpu: bool = False,
    reason: str = "",
    compatible_configs: list[str] | None = None,
    source: str = "",
) -> None:
    """Emit NODE_INCOMPATIBLE_WITH_WORKLOAD when a node has no compatible selfplay configs.

    December 2025: Added to track nodes that are idle but cannot run any selfplay
    due to GPU/memory constraints. This prevents wasted evaluation cycles.

    Args:
        node_id: Identifier for the node
        node_ip: Node IP address
        gpu_vram_gb: GPU VRAM in GB
        has_gpu: Whether node has a GPU
        reason: Why the node is incompatible (e.g., "no_compatible_configs", "gpu_too_small")
        compatible_configs: List of configs node could run (empty if none)
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NODE_INCOMPATIBLE_WITH_WORKLOAD,
        payload={
            "node_id": node_id,
            "node_ip": node_ip,
            "gpu_vram_gb": gpu_vram_gb,
            "has_gpu": has_gpu,
            "reason": reason,
            "compatible_configs": compatible_configs or [],
        },
        source=source,
    ))


async def emit_health_check_passed(
    node_id: str,
    *,
    node_ip: str = "",
    check_type: str = "general",
    latency_ms: float | None = None,
    source: str = "",
) -> None:
    """Emit HEALTH_CHECK_PASSED event after successful health check.

    Args:
        node_id: Identifier for the node
        node_ip: Node IP address
        check_type: Type of health check (ssh, p2p, gpu, general)
        latency_ms: Health check latency in milliseconds
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.HEALTH_CHECK_PASSED,
        payload={
            "node_id": node_id,
            "node_ip": node_ip,
            "check_type": check_type,
            "latency_ms": latency_ms,
        },
        source=source,
    ))


async def emit_health_check_failed(
    node_id: str,
    reason: str,
    *,
    node_ip: str = "",
    check_type: str = "general",
    error: str = "",
    source: str = "",
) -> None:
    """Emit HEALTH_CHECK_FAILED event after failed health check.

    Args:
        node_id: Identifier for the node
        reason: Why the health check failed
        node_ip: Node IP address
        check_type: Type of health check (ssh, p2p, gpu, general)
        error: Error message if any
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.HEALTH_CHECK_FAILED,
        payload={
            "node_id": node_id,
            "reason": reason,
            "node_ip": node_ip,
            "check_type": check_type,
            "error": error,
        },
        source=source,
    ))


async def emit_p2p_cluster_healthy(
    healthy_nodes: int,
    node_count: int,
    *,
    source: str = "",
) -> None:
    """Emit P2P_CLUSTER_HEALTHY event when cluster becomes healthy.

    Args:
        healthy_nodes: Number of healthy nodes
        node_count: Total node count
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.P2P_CLUSTER_HEALTHY,
        payload={
            "healthy": True,
            "healthy_nodes": healthy_nodes,
            "node_count": node_count,
        },
        source=source,
    ))


async def emit_p2p_cluster_unhealthy(
    healthy_nodes: int,
    node_count: int,
    *,
    alerts: list[str] | None = None,
    source: str = "",
) -> None:
    """Emit P2P_CLUSTER_UNHEALTHY event when cluster becomes unhealthy.

    Args:
        healthy_nodes: Number of healthy nodes
        node_count: Total node count
        alerts: List of alert messages
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.P2P_CLUSTER_UNHEALTHY,
        payload={
            "healthy": False,
            "healthy_nodes": healthy_nodes,
            "node_count": node_count,
            "alerts": alerts or [],
        },
        source=source,
    ))


async def emit_p2p_restarted(
    *,
    trigger: str = "recovery_daemon",
    restart_count: int = 0,
    previous_state: str = "",
    source: str = "",
) -> None:
    """Emit P2P_RESTARTED event when P2P orchestrator successfully restarts.

    Dec 30, 2025: Added for event subscription resilience. Subscribers can
    use this to ensure they resubscribe after P2P mesh recovery.

    Args:
        trigger: What triggered the restart (recovery_daemon, manual, etc.)
        restart_count: Total number of restarts since daemon start
        previous_state: State before restart (healthy, unhealthy, partitioned)
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.P2P_RESTARTED,
        payload={
            "trigger": trigger,
            "restart_count": restart_count,
            "previous_state": previous_state,
            "timestamp": time.time(),
        },
        source=source,
    ))


# =============================================================================
# Disk Space Events (December 2025)
# =============================================================================


async def emit_disk_space_low(
    host: str,
    usage_percent: float,
    free_gb: float,
    *,
    threshold: float = 70.0,
    source: str = "",
) -> None:
    """Emit DISK_SPACE_LOW event when disk usage exceeds threshold.

    Args:
        host: Hostname/node ID
        usage_percent: Current disk usage percentage (0-100)
        free_gb: Free disk space in GB
        threshold: The threshold that was exceeded
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.DISK_SPACE_LOW,
        payload={
            "host": host,
            "usage_percent": usage_percent,
            "free_gb": free_gb,
            "threshold": threshold,
        },
        source=source,
    ))


# =============================================================================
# Availability/Provisioning Events (December 28, 2025)
# =============================================================================


async def emit_capacity_low(
    current_gpus: int,
    min_gpus: int,
    provider: str = "",
    *,
    source: str = "CapacityPlanner",
) -> None:
    """Emit CAPACITY_LOW event when GPU capacity drops below threshold.

    Args:
        current_gpus: Current number of active GPUs
        min_gpus: Minimum GPU threshold
        provider: Provider with low capacity (optional, empty for cluster-wide)
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CAPACITY_LOW,
        payload={
            "current_gpus": current_gpus,
            "min_gpus": min_gpus,
            "provider": provider,
        },
        source=source,
    ))


async def emit_capacity_restored(
    current_gpus: int,
    min_gpus: int,
    *,
    source: str = "CapacityPlanner",
) -> None:
    """Emit CAPACITY_RESTORED event when capacity is back above threshold.

    Args:
        current_gpus: Current number of active GPUs
        min_gpus: Minimum GPU threshold
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.CAPACITY_RESTORED,
        payload={
            "current_gpus": current_gpus,
            "min_gpus": min_gpus,
        },
        source=source,
    ))


async def emit_node_provisioned(
    node_id: str,
    provider: str,
    gpu_type: str,
    ip_address: str = "",
    *,
    source: str = "Provisioner",
) -> None:
    """Emit NODE_PROVISIONED event when a new node is created.

    Args:
        node_id: Unique identifier for the new node
        provider: Cloud provider name (vast, lambda, runpod, etc.)
        gpu_type: GPU type on the node (e.g., "GH200", "H100")
        ip_address: Node IP address if available
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NODE_PROVISIONED,
        payload={
            "node_id": node_id,
            "provider": provider,
            "gpu_type": gpu_type,
            "ip_address": ip_address,
        },
        source=source,
    ))


async def emit_node_provision_failed(
    provider: str,
    gpu_type: str,
    error: str,
    *,
    source: str = "Provisioner",
) -> None:
    """Emit NODE_PROVISION_FAILED event when node provisioning fails.

    Args:
        provider: Cloud provider that failed
        gpu_type: GPU type requested
        error: Error message
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NODE_PROVISION_FAILED,
        payload={
            "provider": provider,
            "gpu_type": gpu_type,
            "error": error,
        },
        source=source,
    ))


async def emit_budget_exceeded(
    budget_type: str,
    limit: float,
    current: float,
    *,
    source: str = "CapacityPlanner",
) -> None:
    """Emit BUDGET_EXCEEDED event when spending exceeds limit.

    Args:
        budget_type: Type of budget exceeded ("hourly" or "daily")
        limit: Budget limit in USD
        current: Current spending in USD
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.BUDGET_EXCEEDED,
        payload={
            "budget_type": budget_type,
            "limit_usd": limit,
            "current_usd": current,
        },
        source=source,
    ))


async def emit_budget_alert(
    budget_type: str,
    limit: float,
    current: float,
    threshold_percent: float = 80.0,
    *,
    source: str = "CapacityPlanner",
) -> None:
    """Emit BUDGET_ALERT event when approaching budget threshold.

    Args:
        budget_type: Type of budget ("hourly" or "daily")
        limit: Budget limit in USD
        current: Current spending in USD
        threshold_percent: Threshold percentage that triggered alert
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.BUDGET_ALERT,
        payload={
            "budget_type": budget_type,
            "limit_usd": limit,
            "current_usd": current,
            "threshold_percent": threshold_percent,
        },
        source=source,
    ))


# =============================================================================
# Selfplay Coordination Events (December 2025)
# =============================================================================
# P0 fix for autonomous training loop - enables SelfplayScheduler feedback


async def emit_selfplay_rate_changed(
    config_key: str,
    old_rate: float,
    new_rate: float,
    reason: str,
    source: str = "selfplay_scheduler",
) -> None:
    """Emit a SELFPLAY_RATE_CHANGED event when rate multiplier changes >20%.

    Used to notify IdleResourceDaemon and other consumers of significant
    changes to selfplay rate.
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.SELFPLAY_RATE_CHANGED,
        payload={
            "config_key": config_key,
            "old_rate": old_rate,
            "new_rate": new_rate,
            "reason": reason,
        },
        source=source,
    ))


async def emit_selfplay_allocation_updated(
    config_key: str,
    allocation_weights: dict[str, float],
    reason: str,
    exploration_boost: float = 1.0,
    source: str = "selfplay_scheduler",
) -> None:
    """Emit a SELFPLAY_ALLOCATION_UPDATED event when allocation weights change.

    Used to notify IdleResourceDaemon and SelfplayScheduler of changes to
    curriculum weights, exploration boosts, or priority adjustments.
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.SELFPLAY_ALLOCATION_UPDATED,
        payload={
            "config_key": config_key,
            "allocation_weights": allocation_weights,
            "exploration_boost": exploration_boost,
            "reason": reason,
        },
        source=source,
    ))


async def emit_node_capacity_updated(
    node_id: str,
    gpu_memory_gb: float,
    gpu_utilization: float,
    cpu_utilization: float,
    available_slots: int,
    reason: str = "capacity_update",
    source: str = "health_check_orchestrator",
    *,
    queue_depth: int = 0,
) -> None:
    """Emit a NODE_CAPACITY_UPDATED event when node capacity changes.

    Used by SelfplayScheduler and ResourceMonitoringCoordinator to track
    available capacity for job scheduling.

    Sprint 10 (Jan 3, 2026): Unified emitter for NODE_CAPACITY_UPDATED.
    All emission locations should use this function (or its sync variant)
    to ensure consistent payloads.
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.NODE_CAPACITY_UPDATED,
        payload={
            "node_id": node_id,
            "gpu_memory_gb": gpu_memory_gb,
            "gpu_utilization": gpu_utilization,
            "cpu_utilization": cpu_utilization,
            "available_slots": available_slots,
            "queue_depth": queue_depth,
            "reason": reason,
        },
        source=source,
    ))


def emit_node_capacity_updated_sync(
    node_id: str,
    gpu_memory_gb: float = 0.0,
    gpu_utilization: float = 0.0,
    cpu_utilization: float = 0.0,
    available_slots: int = 0,
    reason: str = "capacity_update",
    source: str = "health_check_orchestrator",
    *,
    queue_depth: int = 0,
) -> None:
    """Synchronous version of emit_node_capacity_updated.

    Sprint 10 (Jan 3, 2026): Unified synchronous emitter for NODE_CAPACITY_UPDATED.
    Use this in sync contexts (heartbeat loops, health checks).

    All callers should use this function to ensure consistent payloads:
    - health_check_orchestrator.py
    - sync_router.py
    - p2p_orchestrator.py

    Args:
        node_id: Node identifier
        gpu_memory_gb: Available GPU memory in GB
        gpu_utilization: GPU utilization percentage (0-100)
        cpu_utilization: CPU utilization percentage (0-100)
        available_slots: Number of available job slots
        reason: Reason for capacity update
        source: Source component emitting the event
        queue_depth: Current work queue depth
    """
    try:
        bus = get_event_bus()
        if bus:
            bus.publish_sync(DataEvent(
                event_type=DataEventType.NODE_CAPACITY_UPDATED,
                payload={
                    "node_id": node_id,
                    "gpu_memory_gb": gpu_memory_gb,
                    "gpu_utilization": gpu_utilization,
                    "cpu_utilization": cpu_utilization,
                    "available_slots": available_slots,
                    "queue_depth": queue_depth,
                    "reason": reason,
                },
                source=source,
            ))
    except (AttributeError, RuntimeError):
        # Event bus not available - non-critical
        pass


# =============================================================================
# Parity Validation Events (December 29, 2025)
# =============================================================================


async def emit_parity_failure_rate_changed(
    new_rate: float,
    old_rate: float,
    board_type: str = "",
    num_players: int = 0,
    total_checked: int = 0,
    *,
    source: str = "ParityValidator",
) -> None:
    """Emit PARITY_FAILURE_RATE_CHANGED event when parity failure rate changes significantly.

    Args:
        new_rate: New parity failure rate (0.0-1.0)
        old_rate: Previous parity failure rate (0.0-1.0)
        board_type: Board type being validated (if config-specific)
        num_players: Number of players (if config-specific)
        total_checked: Total games checked in this batch
        source: Component emitting this event
    """
    await get_event_bus().publish(DataEvent(
        event_type=DataEventType.PARITY_FAILURE_RATE_CHANGED,
        payload={
            "new_rate": new_rate,
            "old_rate": old_rate,
            "board_type": board_type,
            "num_players": num_players,
            "total_checked": total_checked,
            "rate_change": new_rate - old_rate,
        },
        source=source,
    ))


# =============================================================================
# Deprecation Support (December 2025)
# =============================================================================
# This module is deprecated in favor of app.coordination.event_router.
# The __getattr__ hook intercepts imports and issues deprecation warnings
# to guide developers to the unified router.

import warnings as _deprecation_warnings

# Track which names have already warned to avoid duplicate warnings
_warned_names: set[str] = set()


def __getattr__(name: str) -> Any:
    """Intercept module imports and issue deprecation warnings.

    This allows us to maintain backward compatibility while guiding users
    to the unified event router in app.coordination.event_router.

    Args:
        name: The name being imported

    Returns:
        The requested object from the module namespace

    Raises:
        AttributeError: If name is not a known export
    """
    # All public exports that should trigger a deprecation warning
    DEPRECATED_EXPORTS = {
        'DataEvent',
        'DataEventType',
        'EventBus',
        'get_event_bus',
        'reset_event_bus',
    }

    if name in DEPRECATED_EXPORTS:
        # Only warn once per name to avoid spam in test output
        if name not in _warned_names:
            _warned_names.add(name)
            _deprecation_warnings.warn(
                f"Direct imports from app.distributed.data_events are deprecated "
                f"(December 2025). Use app.coordination.event_router instead:\n"
                f"    from app.coordination.event_router import {name}",
                DeprecationWarning,
                stacklevel=2,
            )
        # Return the object from this module's globals
        return globals()[name]

    # Raise AttributeError for unknown names (standard Python behavior)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
