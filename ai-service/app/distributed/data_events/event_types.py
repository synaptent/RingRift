"""Data Event Types and Core Data Structures.

This module contains the DataEventType enum and DataEvent dataclass that form
the foundation of the event system. All other event modules depend on these types.

.. deprecated:: December 2025
    This module is being superseded by the unified event router.
    For new code, prefer using:

        from app.coordination.event_router import DataEventType, DataEvent

    This module remains fully functional for backwards compatibility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
    # Automates transfer learning: 2p -> 3p -> 4p when Elo thresholds are met
    CASCADE_TRANSFER_TRIGGERED = "cascade_transfer_triggered"  # Transfer learning triggered between player counts
    CASCADE_TRANSFER_FAILED = "cascade_transfer_failed"  # Transfer learning failed

    # Reanalysis events (January 27, 2026 - Phase 2.1)
    # Re-evaluates historical games with improved models for better training targets
    REANALYSIS_STARTED = "reanalysis_started"  # Reanalysis job started for a config
    REANALYSIS_COMPLETED = "reanalysis_completed"  # Reanalysis job finished successfully
    REANALYSIS_FAILED = "reanalysis_failed"  # Reanalysis job failed

    # Evaluation events
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_PROGRESS = "evaluation_progress"
    EVALUATION_COMPLETED = "evaluation_completed"
    EVALUATION_FAILED = "evaluation_failed"
    EVALUATION_BACKPRESSURE = "evaluation_backpressure"  # Dec 29, 2025: Eval queue backlogged, pause training
    EVALUATION_BACKPRESSURE_RELEASED = "evaluation_backpressure_released"  # Dec 29, 2025: Eval queue drained, resume training
    MODEL_EVALUATION_BLOCKED = "model_evaluation_blocked"  # Dec 2025 Phase 3: Model not distributed for eval
    ELO_UPDATED = "elo_updated"
    # January 12, 2026: Elo recording facade events (from elo_recording.py)
    ELO_RECORDING_FAILED = "elo_recording_failed"  # Elo recording failed (queued to DLQ)
    ELO_VALIDATION_FAILED = "elo_validation_failed"  # Harness/model validation failed
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
    CURRICULUM_RESET_REQUESTED = "curriculum_reset_requested"  # Jan 26, 2026: Reset curriculum on extended stalls (96+ hours)
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
    QUALITY_PENALTY_APPLIED = "quality_penalty_applied"  # Penalty applied -> reduce selfplay rate
    # Jan 5, 2026 - Session 17.29: Quality-triggered auto-dispatch
    QUALITY_SELFPLAY_DISPATCHED = "quality_selfplay_dispatched"  # High-quality selfplay auto-dispatched

    # P2P recovery events
    VOTER_HEALING_REQUESTED = "voter_healing_requested"  # Jan 2026: Voter healing requested by P2PRecoveryDaemon
    # P1.4 Dec 2025: Added orphaned event types to enable type-safe subscriptions
    EXPLORATION_BOOST = "exploration_boost"  # Request to boost exploration temperature
    EXPLORATION_ADJUSTED = "exploration_adjusted"  # Exploration strategy changed (from FeedbackSignals)
    OPPONENT_MASTERED = "opponent_mastered"  # Opponent mastered -> advance curriculum

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
    PEER_DISCOVERY_EMERGENCY = "peer_discovery_emergency"  # Jan 2026: Emergency peer discovery during quorum crisis

    # Partition healing events (January 2026)
    PARTITION_HEALING_STARTED = "partition_healing_started"  # Healing pass initiated
    PARTITION_HEALED = "partition_healed"  # Partitions successfully healed
    PARTITION_HEALING_FAILED = "partition_healing_failed"  # Healing attempt failed
    HEALING_CONVERGENCE_FAILED = "healing_convergence_failed"  # Jan 4, 2026: Healing reported success but gossip didn't converge
    P2P_RECOVERY_NEEDED = "p2p_recovery_needed"  # Jan 3, 2026: Max escalation reached, manual intervention needed
    P2P_ZOMBIE_DETECTED = "p2p_zombie_detected"  # Jan 7, 2026: HTTP server crashed but process continues, terminating
    P2P_LOOP_STARTUP_FAILED = "p2p_loop_startup_failed"  # Jan 7, 2026: P2P background loop failed to start within timeout
    P2P_LOOP_PERFORMANCE_DEGRADED = "p2p_loop_performance_degraded"  # Jan 7, 2026: Loop avg run duration exceeds 50% of interval
    EVENT_LOOP_BLOCKED = "event_loop_blocked"  # Jan 24, 2026: Event loop blocked by sync operation, affects responsiveness
    GOSSIP_STATE_CLEANUP_COMPLETED = "gossip_state_cleanup_completed"  # Jan 7, 2026: Gossip state TTL cleanup completed
    BACKUP_CANDIDATES_PROBED = "backup_candidates_probed"  # Jan 10, 2026: Backup leader candidates health probed

    # Progress monitoring events (December 2025 - 48h autonomous operation)
    PROGRESS_STALL_DETECTED = "progress_stall_detected"  # Config Elo stalled, recovery triggered
    PROGRESS_RECOVERED = "progress_recovered"  # Config resumed making Elo progress

    # Orphan detection events
    ORPHAN_GAMES_DETECTED = "orphan_games_detected"  # Unregistered game databases found
    ORPHAN_GAMES_REGISTERED = "orphan_games_registered"  # Orphans auto-registered

    # Data integrity events (January 2026 - orphan prevention)
    GAME_SAVE_FAILED = "game_save_failed"  # Failed to save game to database (P0 issue)
    ORPHAN_GAME_PREVENTED = "orphan_game_prevented"  # Orphan game prevented by validation

    # Replication repair events (December 2025)
    REPAIR_COMPLETED = "repair_completed"  # Repair job succeeded
    REPAIR_FAILED = "repair_failed"  # Repair job failed
    REPLICATION_ALERT = "replication_alert"  # Replication health alert

    # Database lifecycle events (Phase 4A.3 - December 2025)
    DATABASE_CREATED = "database_created"  # New database file created - immediate registration

    # SQLite connection pooling events (January 2026 - Phase 6 P2P Stability)
    SQLITE_BACKPRESSURE = "sqlite_backpressure"  # Connection limit causing backpressure (timeout waiting for slot)
    SQLITE_CONNECTION_WARNING = "sqlite_connection_warning"  # Approaching connection limit (80% threshold)

    # Transport corruption events (January 2026 - Phase 8 P2P Stability)
    TRANSPORT_CORRUPTION_DETECTED = "transport_corruption_detected"  # Binary transfer corruption detected on transport

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
    PRIVATE_IP_ADVERTISED = "private_ip_advertised"  # Jan 12, 2026: Node advertising private IP instead of Tailscale
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
    # Jan 12, 2026: Tier-specific memory pressure events for MemoryPressureController
    MEMORY_PRESSURE_CAUTION = "memory_pressure_caution"  # 60-70% RAM - log warning, emit event
    MEMORY_PRESSURE_WARNING = "memory_pressure_warning"  # 70-80% RAM - pause selfplay, reduce batch sizes
    MEMORY_PRESSURE_CRITICAL = "memory_pressure_critical"  # 80-90% RAM - kill non-essential daemons, trigger GC
    MEMORY_PRESSURE_EMERGENCY = "memory_pressure_emergency"  # 90%+ RAM - graceful shutdown, notify standby
    MEMORY_PRESSURE_RECOVERED = "memory_pressure_recovered"  # RAM dropped below threshold - resume normal ops
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
    CLUSTER_VISIBILITY_DEGRADED = "cluster_visibility_degraded"  # Jan 2026: Cluster manifest unavailable, using local-only counts
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


__all__ = [
    "DataEventType",
    "DataEvent",
]
