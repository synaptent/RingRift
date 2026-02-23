"""Daemon type definitions and data structures.

Provides the core types used by DaemonManager:
- DaemonType enum - all supported daemon types
- DaemonState enum - daemon lifecycle states
- DaemonInfo dataclass - daemon runtime information
- DaemonManagerConfig dataclass - manager configuration

Extracted from daemon_manager.py to improve modularity (Dec 2025).
"""

from __future__ import annotations

import asyncio
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from app.config.coordination_defaults import DaemonLoopDefaults

__all__ = [
    "CRITICAL_DAEMONS",
    "DAEMON_DEPENDENCIES",
    "DAEMON_STARTUP_ORDER",
    "DAEMON_CATEGORY_MAP",
    "DaemonCategory",
    "DaemonInfo",
    "DaemonManagerConfig",
    "DaemonState",
    "DaemonType",
    "MAX_RESTART_DELAY",
    "DAEMON_RESTART_RESET_AFTER",
    "RestartTier",  # December 2025: Graceful degradation
    "get_daemon_category",
    "get_daemon_startup_position",
    "mark_daemon_ready",
    "register_mark_ready_callback",
    "validate_daemon_dependencies",
    "validate_startup_order_consistency",
    "validate_startup_order_or_raise",
]


# =============================================================================
# Deprecated Daemon Type Tracking (December 2025)
# =============================================================================

# Daemon types scheduled for removal in Q2 2026
_DEPRECATED_DAEMON_TYPES: dict[str, tuple[str, str]] = {
    "sync_coordinator": ("AUTO_SYNC", "Q2 2026"),
    "health_check": ("NODE_HEALTH_MONITOR", "Q2 2026"),
    # December 2025: Added missing deprecated types
    "ephemeral_sync": ("AUTO_SYNC", "Q2 2026"),
    "cluster_data_sync": ("AUTO_SYNC", "Q2 2026"),
    "system_health_monitor": ("HEALTH_SERVER", "Q2 2026"),  # Use unified_health_manager
    # December 2025: Lambda GH200 nodes are dedicated training (restored Dec 28). Use UNIFIED_IDLE for ephemeral nodes.
    "lambda_idle": ("UNIFIED_IDLE", "Q2 2026"),
    # January 2026: Consolidated backup/sync daemons (Session 17.41)
    # S3 sync consolidation: 3 daemons → S3_SYNC
    "s3_backup": ("S3_SYNC", "Q2 2026"),
    "s3_node_sync": ("S3_SYNC", "Q2 2026"),
    "s3_push": ("S3_SYNC", "Q2 2026"),
    # OWC sync consolidation: 4 daemons → OWC_SYNC_MANAGER
    "external_drive_sync": ("OWC_SYNC_MANAGER", "Q2 2026"),
    "owc_push": ("OWC_SYNC_MANAGER", "Q2 2026"),
    "dual_backup": ("OWC_SYNC_MANAGER", "Q2 2026"),
    "unified_backup": ("OWC_SYNC_MANAGER", "Q2 2026"),
}


def _check_deprecated_daemon(daemon_type: "DaemonType") -> None:
    """Emit deprecation warning for deprecated daemon types."""
    if daemon_type.value in _DEPRECATED_DAEMON_TYPES:
        replacement, removal_date = _DEPRECATED_DAEMON_TYPES[daemon_type.value]
        warnings.warn(
            f"DaemonType.{daemon_type.name} is deprecated and will be removed in {removal_date}. "
            f"Use DaemonType.{replacement} instead.",
            DeprecationWarning,
            stacklevel=3,
        )


class DaemonType(Enum):
    """Types of daemons that can be managed."""
    # Sync daemons
    # DEPRECATED (Dec 2025): SYNC_COORDINATOR replaced by AUTO_SYNC - removal Q2 2026
    SYNC_COORDINATOR = "sync_coordinator"
    HIGH_QUALITY_SYNC = "high_quality_sync"
    ELO_SYNC = "elo_sync"
    MODEL_SYNC = "model_sync"

    # Health/monitoring
    # DEPRECATED (Dec 2025): HEALTH_CHECK replaced by NODE_HEALTH_MONITOR - removal Q2 2026
    HEALTH_CHECK = "health_check"
    CLUSTER_MONITOR = "cluster_monitor"
    QUEUE_MONITOR = "queue_monitor"
    # DEPRECATED (Dec 2025): Use UnifiedNodeHealthDaemon (health_check_orchestrator) - removal Q2 2026
    NODE_HEALTH_MONITOR = "node_health_monitor"

    # Event processing
    EVENT_ROUTER = "event_router"
    CROSS_PROCESS_POLLER = "cross_process_poller"
    DLQ_RETRY = "dlq_retry"
    DAEMON_WATCHDOG = "daemon_watchdog"  # Monitors daemon health & restarts

    # Pipeline daemons
    DATA_PIPELINE = "data_pipeline"
    SELFPLAY_COORDINATOR = "selfplay_coordinator"

    # P2P services
    P2P_BACKEND = "p2p_backend"
    GOSSIP_SYNC = "gossip_sync"
    DATA_SERVER = "data_server"

    # Training enhancement daemons (December 2025)
    DISTILLATION = "distillation"
    UNIFIED_PROMOTION = "unified_promotion"
    EXTERNAL_DRIVE_SYNC = "external_drive_sync"
    VAST_CPU_PIPELINE = "vast_cpu_pipeline"

    # Continuous training loop (December 2025)
    CONTINUOUS_TRAINING_LOOP = "continuous_training_loop"

    # DEPRECATED (Dec 2025): Use AutoSyncDaemon(strategy="broadcast") - removal Q2 2026
    CLUSTER_DATA_SYNC = "cluster_data_sync"

    # Model distribution (December 2025) - auto-distribute models after promotion
    MODEL_DISTRIBUTION = "model_distribution"

    # Automated P2P data sync (December 2025)
    AUTO_SYNC = "auto_sync"

    # Config sync daemon (January 2026) - auto-sync distributed_hosts.yaml across cluster
    # Coordinator detects changes via mtime polling, workers pull on CONFIG_UPDATED event
    # Fixes P2P voter config drift issue where nodes have mismatched voter lists
    CONFIG_SYNC = "config_sync"

    # Config validator daemon (January 2026) - validates config against provider APIs
    # Checks Lambda, Vast, RunPod, Tailscale to detect config drift
    CONFIG_VALIDATOR = "config_validator"

    # Training node watcher (December 2025 - Phase 6)
    TRAINING_NODE_WATCHER = "training_node_watcher"

    # Training data sync (December 2025) - pre-training data sync from OWC/S3
    TRAINING_DATA_SYNC = "training_data_sync"

    # Training data recovery (January 2026 Sprint 13.3) - auto-recover from NPZ corruption
    TRAINING_DATA_RECOVERY = "training_data_recovery"

    # Training watchdog (January 2026 Sprint 17) - monitors training processes for stalls
    # Kills stale processes that haven't sent heartbeats and releases their locks
    TRAINING_WATCHDOG = "training_watchdog"

    # Export watchdog (January 2026 Session 17.41) - monitors export scripts for hangs
    # Kills export_replay_dataset.py processes that exceed max runtime (default 30min)
    EXPORT_WATCHDOG = "export_watchdog"

    # OWC external drive import (December 2025) - periodic import from OWC drive on mac-studio
    # Imports training data from external archive drive for underserved configs
    OWC_IMPORT = "owc_import"

    # OWC model import (Sprint 13 Session 4 - Jan 3, 2026)
    # Imports MODEL FILES (not databases) from OWC drive for Elo evaluation
    # OWC has 1000s of trained models that have never been evaluated
    OWC_MODEL_IMPORT = "owc_model_import"

    # Production game import (January 2026)
    # Imports human vs AI games from ringrift.ai production server
    PRODUCTION_GAME_IMPORT = "production_game_import"

    # Unevaluated model scanner (Sprint 13 Session 4 - Jan 3, 2026)
    # Scans all model sources (local, OWC, cluster, registry) for models without Elo ratings
    # Queues them for evaluation with curriculum-aware priority
    UNEVALUATED_MODEL_SCANNER = "unevaluated_model_scanner"

    # Stale evaluation daemon (Sprint 13 Session 4 - Jan 3, 2026)
    # Re-evaluates models with old Elo ratings (>30 days) to maintain accuracy
    # Runs at lower priority than new model evaluation
    STALE_EVALUATION = "stale_evaluation"

    # Comprehensive model scan daemon (Sprint 17.9 - Jan 9, 2026)
    # Discovers ALL models across local, cluster, and OWC sources
    # Queues each model+harness combination for evaluation
    # Ensures all models get fresh Elo ratings under multiple harnesses
    COMPREHENSIVE_MODEL_SCAN = "comprehensive_model_scan"

    # Backlog evaluation daemon (Sprint 15 - Jan 3, 2026)
    # Discovers models on OWC drive and queues them for Elo evaluation
    # Rate-limited, respects backpressure from EvaluationDaemon
    BACKLOG_EVALUATION = "backlog_evaluation"

    # DEPRECATED (Dec 2025): Use AutoSyncDaemon(strategy="ephemeral") - removal Q2 2026
    EPHEMERAL_SYNC = "ephemeral_sync"

    # P2P auto-deployment (December 2025) - ensure P2P runs on all nodes
    P2P_AUTO_DEPLOY = "p2p_auto_deploy"

    # Replication monitor (December 2025) - monitor data replication health
    REPLICATION_MONITOR = "replication_monitor"

    # Replication repair (December 2025) - actively repair under-replicated data
    REPLICATION_REPAIR = "replication_repair"

    # Tournament daemon (December 2025) - automatic tournament scheduling
    TOURNAMENT_DAEMON = "tournament_daemon"

    # Feedback loop controller (December 2025) - orchestrates all feedback signals
    FEEDBACK_LOOP = "feedback_loop"

    # NPZ distribution (December 2025) - sync training data after export
    NPZ_DISTRIBUTION = "npz_distribution"

    # Orphan detection (December 2025) - detect orphaned games not in manifest
    ORPHAN_DETECTION = "orphan_detection"

    # Auto-evaluation (December 2025) - trigger evaluation after training completes
    EVALUATION = "evaluation"

    # Auto-promotion (December 2025) - auto-promote models based on evaluation results
    AUTO_PROMOTION = "auto_promotion"

    # Reanalysis (January 2026 - Phase 2.1) - re-evaluates historical games with stronger models
    # Triggers when model Elo improves by min_elo_delta (default: 50)
    # Expected impact: +25-50 Elo from improved training targets
    REANALYSIS = "reanalysis"

    # S3 backup (December 2025) - backup models to S3 after promotion
    S3_BACKUP = "s3_backup"

    # S3 node sync (December 2025) - bi-directional S3 sync for all cluster nodes
    S3_NODE_SYNC = "s3_node_sync"

    # S3 consolidation (December 2025) - consolidates data from all nodes (coordinator only)
    S3_CONSOLIDATION = "s3_consolidation"

    # Quality monitor (December 2025) - continuous selfplay quality monitoring
    QUALITY_MONITOR = "quality_monitor"

    # Integrity check (December 2025) - scans for games without move data
    INTEGRITY_CHECK = "integrity_check"

    # Model performance watchdog (December 2025) - monitors model win rates
    MODEL_PERFORMANCE_WATCHDOG = "model_performance_watchdog"

    # Job scheduler (December 2025) - centralized job scheduling with PID-based resource allocation
    JOB_SCHEDULER = "job_scheduler"
    RESOURCE_OPTIMIZER = "resource_optimizer"  # Optimizes resource allocation

    # Idle resource daemon (December 2025 - Phase 20) - monitors idle GPUs and spawns selfplay
    IDLE_RESOURCE = "idle_resource"

    # Node recovery daemon (December 2025 - Phase 21) - auto-recovers terminated nodes
    NODE_RECOVERY = "node_recovery"

    # Tailscale health daemon (December 2025) - monitors and auto-recovers Tailscale connectivity
    # Runs on each cluster node to ensure P2P mesh stability
    TAILSCALE_HEALTH = "tailscale_health"

    # Queue populator (December 2025 - Phase 4) - auto-populates work queue with selfplay/training jobs
    QUEUE_POPULATOR = "queue_populator"

    # Work queue monitor (December 2025) - tracks WORK_* lifecycle events
    # Provides queue depth, latency metrics, backpressure signaling, stuck job detection
    WORK_QUEUE_MONITOR = "work_queue_monitor"

    # Coordinator health monitor (December 2025) - tracks COORDINATOR_* lifecycle events
    # Provides coordinator health state, heartbeat monitoring, cluster health summary
    COORDINATOR_HEALTH_MONITOR = "coordinator_health_monitor"

    # Curriculum integration (December 2025) - bridges all feedback loops for self-improvement
    CURRICULUM_INTEGRATION = "curriculum_integration"

    # Auto export (December 2025) - triggers NPZ export when game thresholds met
    AUTO_EXPORT = "auto_export"

    # Training trigger (December 2025) - decides WHEN to trigger training automatically
    TRAINING_TRIGGER = "training_trigger"

    # Gauntlet feedback controller (December 2025) - bridges gauntlet evaluation to training feedback
    GAUNTLET_FEEDBACK = "gauntlet_feedback"

    # Recovery orchestrator (December 2025) - handles model/training state recovery
    RECOVERY_ORCHESTRATOR = "recovery_orchestrator"

    # Cache coordination (December 2025) - coordinates model caching across cluster
    CACHE_COORDINATION = "cache_coordination"

    # Metrics analysis (December 2025) - continuous metrics monitoring and plateau detection
    METRICS_ANALYSIS = "metrics_analysis"

    # PER orchestrator (December 2025) - monitors Prioritized Experience Replay buffers
    PER_ORCHESTRATOR = "per_orchestrator"

    # Adaptive resource manager (December 2025) - dynamic resource scaling based on workload
    ADAPTIVE_RESOURCES = "adaptive_resources"

    # Multi-provider orchestrator (December 2025) - coordinates across Vast/RunPod/Nebius/etc
    MULTI_PROVIDER = "multi_provider"

    # DEPRECATED (Dec 2025): Use unified_health_manager.get_system_health_score() - removal Q2 2026
    SYSTEM_HEALTH_MONITOR = "system_health_monitor"

    # Health server (December 2025) - exposes /health, /ready, /metrics HTTP endpoints
    HEALTH_SERVER = "health_server"

    # Maintenance daemon (December 2025) - log rotation, DB vacuum, cleanup
    MAINTENANCE = "maintenance"

    # Utilization optimizer (December 2025) - optimizes cluster workloads
    # Stops CPU selfplay on GPU nodes, spawns appropriate workloads by provider
    UTILIZATION_OPTIMIZER = "utilization_optimizer"

    # Cluster utilization watchdog (December 30, 2025) - monitors GPU utilization
    # Emits CLUSTER_UNDERUTILIZED when too many GPUs are idle, triggers remediation
    CLUSTER_UTILIZATION_WATCHDOG = "cluster_utilization_watchdog"

    # Lambda idle shutdown (December 2025) - terminates idle Lambda nodes to save costs
    # DEPRECATED: Lambda Labs GH200 nodes are now dedicated training infrastructure (restored Dec 28, 2025).
    # Dedicated GPU nodes don't need idle shutdown - they run continuous training workloads.
    # Use UNIFIED_IDLE for ephemeral/on-demand instances instead.
    LAMBDA_IDLE = "lambda_idle"

    # Vast.ai idle shutdown (December 2025) - terminates idle Vast.ai nodes to save costs
    # Important for ephemeral marketplace instances with hourly billing
    VAST_IDLE = "vast_idle"

    # Cluster watchdog (December 2025) - self-healing cluster utilization monitor
    CLUSTER_WATCHDOG = "cluster_watchdog"

    # Data cleanup (December 2025) - auto-quarantine/delete poor quality databases
    DATA_CLEANUP = "data_cleanup"

    # Data consolidation (December 2025) - consolidate scattered selfplay games into canonical DBs
    DATA_CONSOLIDATION = "data_consolidation"

    # NPZ combination (December 2025) - quality-weighted NPZ combination for training
    # Combines historical data with fresh data using quality and freshness weighting
    NPZ_COMBINATION = "npz_combination"

    # Comprehensive consolidation (January 2026) - scheduled sweep consolidation
    # Unlike DATA_CONSOLIDATION (event-driven), this runs on a schedule to find
    # ALL games across 14+ storage patterns (owc_imports, synced, p2p_gpu, etc.)
    # Ensures no games are missed by the event-driven consolidation daemon
    COMPREHENSIVE_CONSOLIDATION = "comprehensive_consolidation"

    # Disk space manager (December 2025) - proactive disk space management
    # Monitors disk usage and triggers cleanup before reaching critical thresholds
    DISK_SPACE_MANAGER = "disk_space_manager"

    # Coordinator disk manager (December 27, 2025) - disk management for coordinator-only nodes
    # Syncs data to remote storage (OWC) before cleanup, more aggressive thresholds
    COORDINATOR_DISK_MANAGER = "coordinator_disk_manager"

    # Sync push daemon (December 28, 2025) - push-based sync for GPU training nodes
    # GPU nodes proactively push data to coordinator before disk fills
    # Part of "sync then clean" pattern - never delete without verified receipt
    SYNC_PUSH = "sync_push"

    # Unified data plane daemon (December 28, 2025) - consolidated data synchronization
    # Replaces fragmented sync infrastructure (~4,514 LOC consolidated)
    # Components: DataCatalog, SyncPlanner v2, TransportManager, EventBridge
    UNIFIED_DATA_PLANE = "unified_data_plane"

    # Node availability daemon (December 28, 2025) - syncs provider instance state with YAML config
    # Queries cloud provider APIs (Vast, Lambda, RunPod) and updates distributed_hosts.yaml
    # Solves the problem of stale config where nodes are marked 'ready' but actually terminated
    NODE_AVAILABILITY = "node_availability"

    # =========================================================================
    # Cluster Availability Manager daemons (December 28, 2025)
    # Provides automated cluster availability management:
    # - NodeMonitor: Multi-layer health checking (P2P, SSH, GPU, Provider API)
    # - RecoveryEngine: Escalating recovery strategies
    # - Provisioner: Auto-provision new instances when capacity drops
    # - CapacityPlanner: Budget-aware capacity management
    # =========================================================================
    AVAILABILITY_NODE_MONITOR = "availability_node_monitor"
    AVAILABILITY_RECOVERY_ENGINE = "availability_recovery_engine"
    AVAILABILITY_PROVISIONER = "availability_provisioner"
    AVAILABILITY_CAPACITY_PLANNER = "availability_capacity_planner"

    # Cascade training orchestrator (December 29, 2025) - multiplayer bootstrapping
    # Orchestrates 2p → 3p → 4p training cascade with transfer learning
    # Accelerates multiplayer model training by starting from learned features
    CASCADE_TRAINING = "cascade_training"

    # =========================================================================
    # 48-Hour Autonomous Operation daemons (December 29, 2025)
    # These daemons enable the system to run unattended for 48+ hours:
    # - ProgressWatchdog: Detects Elo velocity stalls and triggers recovery
    # - P2PRecovery: Auto-restarts P2P orchestrator on partition/failure
    # - VoterHealthMonitor: Continuous voter probing with multi-transport fallback
    # - MemoryMonitor: Proactive VRAM/RSS monitoring to prevent OOM crashes
    # - StaleFallback: Uses older models when sync fails to maintain selfplay
    # =========================================================================
    PROGRESS_WATCHDOG = "progress_watchdog"
    P2P_RECOVERY = "p2p_recovery"
    VOTER_HEALTH_MONITOR = "voter_health_monitor"  # Dec 30, 2025: Continuous voter probing
    MEMORY_MONITOR = "memory_monitor"
    SOCKET_LEAK_RECOVERY = "socket_leak_recovery"  # Jan 2026: Socket/FD leak detection and recovery
    STALE_FALLBACK = "stale_fallback"
    UNDERUTILIZATION_RECOVERY = "underutilization_recovery"  # Jan 4, 2026: Phase 3 P2P Resilience
    FAST_FAILURE_DETECTOR = "fast_failure_detector"  # Jan 4, 2026: Phase 4 P2P Resilience

    # =========================================================================
    # Connectivity Recovery (December 29, 2025)
    # Unified event-driven connectivity recovery coordinator
    # Handles: TAILSCALE_DISCONNECTED, P2P_NODE_DEAD, HOST_OFFLINE events
    # Bridges TailscaleHealthDaemon, NodeAvailabilityDaemon, P2P orchestrator
    # =========================================================================
    CONNECTIVITY_RECOVERY = "connectivity_recovery"

    # =========================================================================
    # NNUE Automatic Training (December 29, 2025)
    # Automatically trains NNUE models when game thresholds are met
    # Per-config game thresholds: hex8_2p=5000, hex8_4p=10000, square19_2p=2000
    # =========================================================================
    NNUE_TRAINING = "nnue_training"
    ARCHITECTURE_FEEDBACK = "architecture_feedback"  # Dec 29, 2025: Architecture allocation weights

    # =========================================================================
    # Parity Validation Daemon (December 30, 2025)
    # Runs on coordinator (has Node.js) to validate TS/Python parity for
    # canonical databases. Cluster nodes generate databases with "pending_gate"
    # status because they lack npx. This daemon validates them and stores
    # TS reference hashes, enabling hash-based validation on cluster nodes.
    # =========================================================================
    PARITY_VALIDATION = "parity_validation"

    # =========================================================================
    # Elo Progress Tracking (December 31, 2025)
    # =========================================================================
    # Periodically snapshots best model Elo for each config to track
    # improvement over time. Provides evidence of training loop effectiveness.
    # =========================================================================
    ELO_PROGRESS = "elo_progress"

    # =========================================================================
    # Unified Backup Daemon (January 2026)
    # =========================================================================
    # Backs up ALL selfplay games to OWC external drive and AWS S3.
    # Discovers games from all storage patterns via GameDiscovery.
    # Event-driven: responds to DATA_SYNC_COMPLETED, SELFPLAY_COMPLETE.
    # =========================================================================
    UNIFIED_BACKUP = "unified_backup"

    # =========================================================================
    # S3 Push Daemon (January 2026)
    # =========================================================================
    # Pushes all game databases, training NPZ files, and models to S3.
    # Periodically checks for modified files and uploads only changes.
    # Event-driven: responds to DATA_SYNC_COMPLETED, TRAINING_COMPLETED.
    # =========================================================================
    S3_PUSH = "s3_push"

    # =========================================================================
    # Consolidated S3 Sync Daemon (January 2026)
    # =========================================================================
    # Unified S3 sync daemon replacing S3_BACKUP, S3_PUSH, S3_NODE_SYNC.
    # Provides bidirectional S3 sync with intelligent scheduling.
    # =========================================================================
    S3_SYNC = "s3_sync"

    # =========================================================================
    # Cluster Consolidation Daemon (January 2026)
    # =========================================================================
    # CRITICAL: Bridges distributed selfplay with training pipeline.
    # Pulls selfplay games from 30+ cluster nodes into canonical databases.
    # Fixes the handoff failure where games stay on cluster nodes and never
    # reach the coordinator for training.
    # Event-driven: responds to NEW_GAMES_AVAILABLE, SELFPLAY_COMPLETE.
    # =========================================================================
    CLUSTER_CONSOLIDATION = "cluster_consolidation"

    # =========================================================================
    # Unified Data Sync Orchestrator (January 2026)
    # =========================================================================
    # Central coordinator for all data synchronization operations.
    # Listens to DATA_SYNC_COMPLETED events and triggers S3/OWC backups.
    # Tracks replication status and provides unified data visibility.
    # Event-driven: responds to DATA_SYNC_COMPLETED, BACKUP_COMPLETED.
    # =========================================================================
    UNIFIED_DATA_SYNC_ORCHESTRATOR = "unified_data_sync_orchestrator"

    # =========================================================================
    # Comprehensive Data Consolidation System (January 2026)
    # =========================================================================
    # These daemons provide unified data management across all storage:
    # - OWC_PUSH: Push data to OWC external drive for backup
    # - S3_IMPORT: Import data from S3 for recovery/bootstrap
    # - UNIFIED_DATA_CATALOG: Single API for querying data across sources
    # - DUAL_BACKUP: Ensures data is backed up to BOTH S3 AND OWC
    # - NODE_DATA_AGENT: Per-node agent for data discovery and fetching
    # =========================================================================
    OWC_PUSH = "owc_push"
    S3_IMPORT = "s3_import"
    UNIFIED_DATA_CATALOG = "unified_data_catalog"
    DUAL_BACKUP = "dual_backup"
    NODE_DATA_AGENT = "node_data_agent"

    # =========================================================================
    # Consolidated OWC Sync Manager (January 2026)
    # =========================================================================
    # Unified OWC sync manager replacing EXTERNAL_DRIVE_SYNC, OWC_PUSH,
    # DUAL_BACKUP, UNIFIED_BACKUP. Provides intelligent OWC drive sync.
    # =========================================================================
    OWC_SYNC_MANAGER = "owc_sync_manager"

    # =========================================================================
    # Online Model Merge Daemon (January 2026)
    # =========================================================================
    # Validates and merges shadow models (from online learning) into canonical.
    # Part of human game training pipeline - learns from human wins against AI.
    # Runs mini-gauntlet before merge to ensure shadow model improves quality.
    # =========================================================================
    ONLINE_MERGE = "online_merge"

    # =========================================================================
    # Pipeline Completeness Monitor (February 2026)
    # =========================================================================
    # Tracks last completion timestamp for each pipeline stage per config.
    # Emits PIPELINE_STAGE_OVERDUE events when stages exceed thresholds.
    # Health check returns RED if any config has 2+ overdue stages.
    # =========================================================================
    PIPELINE_COMPLETENESS_MONITOR = "pipeline_completeness_monitor"


class DaemonState(Enum):
    """State of a daemon."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    RESTARTING = "restarting"
    IMPORT_FAILED = "import_failed"  # Permanent failure due to missing imports
    DEGRADED = "degraded"  # December 2025: Daemon in degraded mode (exceeds restart limits)


class RestartTier(Enum):
    """Restart tier for graceful degradation.

    December 2025: Part of 48-hour autonomous operation plan.
    Instead of blocking daemons for 24 hours when restart limits are exceeded,
    we use tiered restart policies with degraded mode.

    Tiers:
    - NORMAL: 1-5 restarts, standard exponential backoff (5s → 80s)
    - ELEVATED: 6-10 restarts, extended backoff (160s → 320s)
    - DEGRADED: >10 restarts, keep retrying with longer intervals
      - Critical daemons: 30 min retry interval
      - Non-critical daemons: 4 hour retry interval
    """
    NORMAL = "normal"
    ELEVATED = "elevated"
    DEGRADED = "degraded"


class DaemonCategory(Enum):
    """Daemon categories for hierarchical circuit breaking.

    December 30, 2025: Added for per-category cascade circuit breakers.
    Each category has independent thresholds and cooldowns, allowing:
    - Critical categories (EVENT, PIPELINE) to restart even when others are blocked
    - Isolation of failure cascades within categories
    - Tuned thresholds per category based on expected restart patterns

    Categories:
    - EVENT: Core event routing daemons (EVENT_ROUTER, CROSS_PROCESS_POLLER, DLQ_RETRY)
    - SYNC: Data synchronization daemons (AUTO_SYNC, ELO_SYNC, GOSSIP_SYNC)
    - PIPELINE: Data pipeline daemons (DATA_PIPELINE, SELFPLAY_COORDINATOR)
    - HEALTH: Health monitoring daemons (NODE_HEALTH_MONITOR, QUALITY_MONITOR)
    - EVALUATION: Model evaluation daemons (EVALUATION, AUTO_PROMOTION)
    - DISTRIBUTION: Model/data distribution daemons
    - RESOURCE: Resource management daemons (IDLE_RESOURCE, NODE_RECOVERY)
    - FEEDBACK: Training feedback daemons (FEEDBACK_LOOP, CURRICULUM_INTEGRATION)
    - QUEUE: Queue management daemons (QUEUE_POPULATOR, WORK_QUEUE_MONITOR)
    - RECOVERY: Recovery daemons (RECOVERY_ORCHESTRATOR, CONNECTIVITY_RECOVERY)
    - AUTONOMOUS: Autonomous operation daemons (PROGRESS_WATCHDOG, MEMORY_MONITOR)
    - PROVIDER: Cloud provider daemons (MULTI_PROVIDER, VAST_IDLE)
    - MISC: Miscellaneous daemons not in other categories
    """
    EVENT = "event"
    SYNC = "sync"
    PIPELINE = "pipeline"
    HEALTH = "health"
    EVALUATION = "evaluation"
    DISTRIBUTION = "distribution"
    RESOURCE = "resource"
    FEEDBACK = "feedback"
    QUEUE = "queue"
    RECOVERY = "recovery"
    AUTONOMOUS = "autonomous"
    PROVIDER = "provider"
    MISC = "misc"


# Constants for recovery behavior (December 2025: imported from centralized thresholds)
try:
    from app.config.thresholds import (
        DAEMON_RESTART_DELAY_MAX,
        DAEMON_RESTART_RESET_AFTER,
    )
    MAX_RESTART_DELAY = DAEMON_RESTART_DELAY_MAX  # Legacy alias
except ImportError:
    # Fallback if thresholds not available (see app.config.timeout_config)
    try:
        from app.config.timeout_config import TIMEOUTS
        DAEMON_RESTART_RESET_AFTER = TIMEOUTS.RESTART_RESET_AFTER
        MAX_RESTART_DELAY = TIMEOUTS.MAX_RESTART_DELAY
    except ImportError:
        DAEMON_RESTART_RESET_AFTER = 3600  # Reset restart count after 1 hour of stability
        MAX_RESTART_DELAY = 300  # Cap exponential backoff at 5 minutes


@dataclass
class DaemonInfo:
    """Information about a registered daemon."""
    daemon_type: DaemonType
    state: DaemonState = DaemonState.STOPPED
    task: asyncio.Task | None = None
    start_time: float = 0.0
    restart_count: int = 0
    last_error: str | None = None
    health_check_interval: float = 60.0
    auto_restart: bool = True
    max_restarts: int = 5
    restart_delay: float = 5.0

    # December 2025: Startup grace period before health checks begin
    # Slow-starting daemons (e.g., loading large state files) won't be
    # incorrectly marked as unhealthy during initialization
    startup_grace_period: float = 60.0

    # Dependencies
    depends_on: list[DaemonType] = field(default_factory=list)

    # Jan 2, 2026: Soft dependencies that allow degraded startup
    soft_depends_on: list[DaemonType] = field(default_factory=list)
    startup_mode: str = "degraded"  # "strict", "degraded", or "local"

    # Degraded mode tracking
    missing_soft_deps: list[DaemonType] = field(default_factory=list)
    degraded_mode: bool = False

    # Stability tracking for restart count reset
    stable_since: float = 0.0  # When daemon became stable (no errors)
    last_failure_time: float = 0.0  # When the last failure occurred

    # Import error tracking
    import_error: str | None = None  # Specific import error message

    # P0.3 Dec 2025: Readiness signal to prevent race conditions
    # When daemons are started, they set ready_event when initialization completes
    ready_event: asyncio.Event | None = None
    ready_timeout: float = 30.0  # Max time to wait for daemon to be ready

    # December 2025: Store daemon instance for health_check() calls
    # Set by factory functions in daemon_runners.py
    instance: Any | None = None

    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        if self.state == DaemonState.RUNNING and self.start_time > 0:
            return time.time() - self.start_time
        return 0.0


@dataclass
class DaemonManagerConfig:
    """Configuration for DaemonManager.

    Default values sourced from app.config.coordination_defaults.DaemonLoopDefaults
    for centralized configuration management.
    """

    auto_start: bool = False  # Auto-start all daemons on init

    # Test/embedded mode knobs
    # -------------------------------------------------------------------------
    # Some unit/integration tests register ad-hoc daemon factories that don't
    # need (and shouldn't pay for) the full coordination bootstrap.
    enable_coordination_wiring: bool = True

    # Dependency wait behavior (Dec 27, 2025)
    # start(wait_for_deps=True) will poll these dependencies until ready.
    # Jan 3, 2026: Increased from 30s to 60s to accommodate slow DATA_PIPELINE startup
    dependency_wait_timeout: float = 60.0
    dependency_poll_interval: float = 0.5

    # Dec 2025: Use centralized defaults from coordination_defaults.py
    health_check_interval: float = field(
        default_factory=lambda: float(DaemonLoopDefaults.CHECK_INTERVAL) / 10.0  # 30s (10% of check interval)
    )
    shutdown_timeout: float = field(
        default_factory=lambda: DaemonLoopDefaults.SHUTDOWN_GRACE_PERIOD
    )
    force_kill_timeout: float = field(
        default_factory=lambda: DaemonLoopDefaults.HEALTH_CHECK_TIMEOUT
    )
    auto_restart_failed: bool = True  # Auto-restart failed daemons

    # Registry validation (Dec 28, 2025)
    # -------------------------------------------------------------------------
    # When True, raise ValueError if registry validation fails at startup.
    # When False (default), only log errors and continue with partial registry.
    strict_registry_validation: bool = False
    max_restart_attempts: int = field(
        default_factory=lambda: DaemonLoopDefaults.MAX_CONSECUTIVE_ERRORS
    )
    recovery_cooldown: float = field(
        default_factory=lambda: DaemonLoopDefaults.ERROR_BACKOFF_BASE * 2  # 10s (2x base backoff)
    )
    # P11-HIGH-2: Faster health checks for critical daemons
    critical_daemon_health_interval: float = field(
        default_factory=lambda: DaemonLoopDefaults.HEALTH_CHECK_TIMEOUT * 3  # 15s (3x health timeout)
    )

    # December 2025: Default startup grace period before health checks begin
    # Slow-starting daemons won't be incorrectly marked as unhealthy during initialization
    default_startup_grace_period: float = 60.0  # seconds


# P11-HIGH-2: Daemons critical for cluster health that need faster failure detection
# NOTE (Dec 2025): Only include daemons that are ACTUALLY used in standard profile.
# Optional daemons like GOSSIP_SYNC, DATA_SERVER, EPHEMERAL_SYNC should not be marked
# critical since they're not started by default.
CRITICAL_DAEMONS: set[DaemonType] = {
    DaemonType.EVENT_ROUTER,  # Core event bus - all coordination depends on this
    DaemonType.DAEMON_WATCHDOG,  # Self-healing for daemon crashes (Dec 2025 fix)
    DaemonType.DATA_PIPELINE,  # Pipeline processor - must start before AUTO_SYNC (Dec 2025 fix)
    DaemonType.AUTO_SYNC,  # Primary data sync mechanism
    DaemonType.QUEUE_POPULATOR,  # Keeps work queue populated
    DaemonType.IDLE_RESOURCE,  # Ensures GPUs stay utilized
    DaemonType.FEEDBACK_LOOP,  # Coordinates training feedback signals
    DaemonType.MEMORY_MONITOR,  # Prevents OOM crashes (Dec 30, 2025)
    DaemonType.SOCKET_LEAK_RECOVERY,  # Prevents socket/FD leaks (Jan 2026)
}


# =============================================================================
# Daemon Category Mapping (December 30, 2025)
# =============================================================================
# Maps each DaemonType to its DaemonCategory for hierarchical circuit breaking.
# Categories enable per-group circuit breakers with independent thresholds.
DAEMON_CATEGORY_MAP: dict[DaemonType, DaemonCategory] = {
    # EVENT category - core event routing (high threshold, short cooldown)
    DaemonType.EVENT_ROUTER: DaemonCategory.EVENT,
    DaemonType.CROSS_PROCESS_POLLER: DaemonCategory.EVENT,
    DaemonType.DLQ_RETRY: DaemonCategory.EVENT,
    DaemonType.DAEMON_WATCHDOG: DaemonCategory.EVENT,

    # SYNC category - data synchronization
    DaemonType.SYNC_COORDINATOR: DaemonCategory.SYNC,
    DaemonType.AUTO_SYNC: DaemonCategory.SYNC,
    DaemonType.HIGH_QUALITY_SYNC: DaemonCategory.SYNC,
    DaemonType.ELO_SYNC: DaemonCategory.SYNC,
    DaemonType.MODEL_SYNC: DaemonCategory.SYNC,
    DaemonType.GOSSIP_SYNC: DaemonCategory.SYNC,
    DaemonType.CLUSTER_DATA_SYNC: DaemonCategory.SYNC,
    DaemonType.EPHEMERAL_SYNC: DaemonCategory.SYNC,
    DaemonType.EXTERNAL_DRIVE_SYNC: DaemonCategory.SYNC,
    DaemonType.TRAINING_DATA_SYNC: DaemonCategory.SYNC,
    DaemonType.OWC_IMPORT: DaemonCategory.SYNC,
    DaemonType.SYNC_PUSH: DaemonCategory.SYNC,
    DaemonType.CLUSTER_CONSOLIDATION: DaemonCategory.SYNC,  # Jan 2026: Pull games from cluster
    DaemonType.UNIFIED_DATA_SYNC_ORCHESTRATOR: DaemonCategory.SYNC,  # Jan 2026: Coordinate all data sync
    DaemonType.OWC_PUSH: DaemonCategory.SYNC,  # Jan 2026: Push to OWC external drive
    DaemonType.S3_IMPORT: DaemonCategory.SYNC,  # Jan 2026: Import from S3
    DaemonType.UNIFIED_DATA_CATALOG: DaemonCategory.SYNC,  # Jan 2026: Unified data catalog API
    DaemonType.DUAL_BACKUP: DaemonCategory.SYNC,  # Jan 2026: Dual S3+OWC backup
    DaemonType.NODE_DATA_AGENT: DaemonCategory.DISTRIBUTION,  # Jan 2026: Per-node data agent
    DaemonType.S3_SYNC: DaemonCategory.SYNC,  # Jan 2026: Consolidated S3 sync (replaces S3_BACKUP, S3_PUSH, S3_NODE_SYNC)
    DaemonType.OWC_SYNC_MANAGER: DaemonCategory.SYNC,  # Jan 2026: Consolidated OWC sync (replaces EXTERNAL_DRIVE_SYNC, OWC_PUSH, DUAL_BACKUP)

    # PIPELINE category - data pipeline (high threshold, exempt from global)
    DaemonType.DATA_PIPELINE: DaemonCategory.PIPELINE,
    DaemonType.SELFPLAY_COORDINATOR: DaemonCategory.PIPELINE,
    DaemonType.AUTO_EXPORT: DaemonCategory.PIPELINE,
    DaemonType.DATA_CONSOLIDATION: DaemonCategory.PIPELINE,
    DaemonType.NPZ_COMBINATION: DaemonCategory.PIPELINE,
    DaemonType.COMPREHENSIVE_CONSOLIDATION: DaemonCategory.PIPELINE,  # Jan 2026: Scheduled sweep
    DaemonType.CONTINUOUS_TRAINING_LOOP: DaemonCategory.PIPELINE,
    DaemonType.CASCADE_TRAINING: DaemonCategory.PIPELINE,

    # HEALTH category - health monitoring
    DaemonType.HEALTH_CHECK: DaemonCategory.HEALTH,
    DaemonType.CLUSTER_MONITOR: DaemonCategory.HEALTH,
    DaemonType.QUEUE_MONITOR: DaemonCategory.HEALTH,
    DaemonType.NODE_HEALTH_MONITOR: DaemonCategory.HEALTH,
    DaemonType.QUALITY_MONITOR: DaemonCategory.HEALTH,
    DaemonType.MODEL_PERFORMANCE_WATCHDOG: DaemonCategory.HEALTH,
    DaemonType.HEALTH_SERVER: DaemonCategory.HEALTH,
    DaemonType.COORDINATOR_HEALTH_MONITOR: DaemonCategory.HEALTH,
    DaemonType.INTEGRITY_CHECK: DaemonCategory.HEALTH,
    DaemonType.CLUSTER_WATCHDOG: DaemonCategory.HEALTH,
    DaemonType.CLUSTER_UTILIZATION_WATCHDOG: DaemonCategory.HEALTH,

    # EVALUATION category - model evaluation
    DaemonType.EVALUATION: DaemonCategory.EVALUATION,
    DaemonType.AUTO_PROMOTION: DaemonCategory.EVALUATION,
    DaemonType.UNIFIED_PROMOTION: DaemonCategory.EVALUATION,
    DaemonType.TOURNAMENT_DAEMON: DaemonCategory.EVALUATION,
    DaemonType.GAUNTLET_FEEDBACK: DaemonCategory.EVALUATION,
    DaemonType.BACKLOG_EVALUATION: DaemonCategory.EVALUATION,  # Sprint 15
    DaemonType.COMPREHENSIVE_MODEL_SCAN: DaemonCategory.EVALUATION,  # Sprint 17.9: Multi-harness scan

    # DISTRIBUTION category - model/data distribution
    DaemonType.MODEL_DISTRIBUTION: DaemonCategory.DISTRIBUTION,
    DaemonType.NPZ_DISTRIBUTION: DaemonCategory.DISTRIBUTION,
    DaemonType.REPLICATION_MONITOR: DaemonCategory.DISTRIBUTION,
    DaemonType.REPLICATION_REPAIR: DaemonCategory.DISTRIBUTION,
    DaemonType.S3_BACKUP: DaemonCategory.DISTRIBUTION,
    DaemonType.S3_NODE_SYNC: DaemonCategory.DISTRIBUTION,
    DaemonType.S3_CONSOLIDATION: DaemonCategory.DISTRIBUTION,
    DaemonType.UNIFIED_DATA_PLANE: DaemonCategory.DISTRIBUTION,
    DaemonType.UNIFIED_BACKUP: DaemonCategory.DISTRIBUTION,  # Jan 2026: OWC + S3 backup
    DaemonType.S3_PUSH: DaemonCategory.DISTRIBUTION,  # Jan 2026: S3 backup push

    # RESOURCE category - resource management
    DaemonType.IDLE_RESOURCE: DaemonCategory.RESOURCE,
    DaemonType.NODE_RECOVERY: DaemonCategory.RESOURCE,
    DaemonType.JOB_SCHEDULER: DaemonCategory.RESOURCE,
    DaemonType.RESOURCE_OPTIMIZER: DaemonCategory.RESOURCE,
    DaemonType.ADAPTIVE_RESOURCES: DaemonCategory.RESOURCE,
    DaemonType.UTILIZATION_OPTIMIZER: DaemonCategory.RESOURCE,
    DaemonType.DISK_SPACE_MANAGER: DaemonCategory.RESOURCE,
    DaemonType.COORDINATOR_DISK_MANAGER: DaemonCategory.RESOURCE,
    DaemonType.DATA_CLEANUP: DaemonCategory.RESOURCE,

    # FEEDBACK category - training feedback (high threshold, exempt from global)
    DaemonType.FEEDBACK_LOOP: DaemonCategory.FEEDBACK,
    DaemonType.CURRICULUM_INTEGRATION: DaemonCategory.FEEDBACK,
    DaemonType.ARCHITECTURE_FEEDBACK: DaemonCategory.FEEDBACK,
    DaemonType.TRAINING_TRIGGER: DaemonCategory.FEEDBACK,
    DaemonType.DISTILLATION: DaemonCategory.FEEDBACK,
    DaemonType.NNUE_TRAINING: DaemonCategory.FEEDBACK,

    # QUEUE category - queue management
    DaemonType.QUEUE_POPULATOR: DaemonCategory.QUEUE,
    DaemonType.WORK_QUEUE_MONITOR: DaemonCategory.QUEUE,
    DaemonType.ORPHAN_DETECTION: DaemonCategory.QUEUE,
    DaemonType.TRAINING_NODE_WATCHER: DaemonCategory.QUEUE,

    # RECOVERY category - recovery daemons
    DaemonType.RECOVERY_ORCHESTRATOR: DaemonCategory.RECOVERY,
    DaemonType.CONNECTIVITY_RECOVERY: DaemonCategory.RECOVERY,
    DaemonType.TAILSCALE_HEALTH: DaemonCategory.RECOVERY,
    DaemonType.P2P_AUTO_DEPLOY: DaemonCategory.RECOVERY,
    DaemonType.CACHE_COORDINATION: DaemonCategory.RECOVERY,
    DaemonType.METRICS_ANALYSIS: DaemonCategory.RECOVERY,
    DaemonType.PER_ORCHESTRATOR: DaemonCategory.RECOVERY,

    # AUTONOMOUS category - 48h autonomous operation (high threshold, exempt)
    DaemonType.PROGRESS_WATCHDOG: DaemonCategory.AUTONOMOUS,
    DaemonType.P2P_RECOVERY: DaemonCategory.AUTONOMOUS,
    DaemonType.VOTER_HEALTH_MONITOR: DaemonCategory.AUTONOMOUS,
    DaemonType.MEMORY_MONITOR: DaemonCategory.AUTONOMOUS,
    DaemonType.SOCKET_LEAK_RECOVERY: DaemonCategory.AUTONOMOUS,
    DaemonType.TRAINING_WATCHDOG: DaemonCategory.AUTONOMOUS,  # Jan 4, 2026: Sprint 17
    DaemonType.EXPORT_WATCHDOG: DaemonCategory.AUTONOMOUS,  # Jan 6, 2026: Session 17.41
    DaemonType.STALE_FALLBACK: DaemonCategory.AUTONOMOUS,
    DaemonType.UNDERUTILIZATION_RECOVERY: DaemonCategory.AUTONOMOUS,  # Jan 4, 2026: Phase 3 P2P Resilience
    DaemonType.FAST_FAILURE_DETECTOR: DaemonCategory.AUTONOMOUS,  # Jan 4, 2026: Phase 4 P2P Resilience
    DaemonType.PARITY_VALIDATION: DaemonCategory.AUTONOMOUS,
    DaemonType.ELO_PROGRESS: DaemonCategory.AUTONOMOUS,  # Dec 31, 2025: Tracks Elo improvement
    DaemonType.MAINTENANCE: DaemonCategory.AUTONOMOUS,
    DaemonType.PIPELINE_COMPLETENESS_MONITOR: DaemonCategory.AUTONOMOUS,  # Feb 2026: Pipeline stage tracking

    # PROVIDER category - cloud provider daemons
    DaemonType.MULTI_PROVIDER: DaemonCategory.PROVIDER,
    DaemonType.VAST_IDLE: DaemonCategory.PROVIDER,
    DaemonType.LAMBDA_IDLE: DaemonCategory.PROVIDER,
    DaemonType.VAST_CPU_PIPELINE: DaemonCategory.PROVIDER,
    DaemonType.NODE_AVAILABILITY: DaemonCategory.PROVIDER,
    DaemonType.AVAILABILITY_NODE_MONITOR: DaemonCategory.PROVIDER,
    DaemonType.AVAILABILITY_RECOVERY_ENGINE: DaemonCategory.PROVIDER,
    DaemonType.AVAILABILITY_PROVISIONER: DaemonCategory.PROVIDER,
    DaemonType.AVAILABILITY_CAPACITY_PLANNER: DaemonCategory.PROVIDER,

    # MISC category - miscellaneous (default fallback)
    DaemonType.P2P_BACKEND: DaemonCategory.MISC,
    DaemonType.DATA_SERVER: DaemonCategory.MISC,
    DaemonType.SYSTEM_HEALTH_MONITOR: DaemonCategory.MISC,
}


def get_daemon_category(daemon_type: DaemonType) -> DaemonCategory:
    """Get the category for a daemon type.

    Args:
        daemon_type: The daemon type to look up.

    Returns:
        The DaemonCategory for this daemon, or MISC if not mapped.
    """
    return DAEMON_CATEGORY_MAP.get(daemon_type, DaemonCategory.MISC)


# P0 Critical Fix (Dec 2025): Daemon startup order to prevent race conditions
# DATA_PIPELINE and FEEDBACK_LOOP must start BEFORE AUTO_SYNC to avoid event loss.
# Events emitted by AUTO_SYNC (DATA_SYNC_COMPLETED) need handlers ready.
DAEMON_STARTUP_ORDER: list[DaemonType] = [
    # =========================================================================
    # Core infrastructure (positions 1-4)
    # =========================================================================
    DaemonType.EVENT_ROUTER,           # 1. Event system must be first
    DaemonType.DAEMON_WATCHDOG,        # 2. Self-healing for daemon crashes
    DaemonType.DATA_PIPELINE,          # 3. Pipeline processor (before sync!)
    DaemonType.FEEDBACK_LOOP,          # 4. Training feedback (before sync!)

    # =========================================================================
    # Sync and queue management (positions 5-10)
    # =========================================================================
    DaemonType.AUTO_SYNC,              # 5. Data sync (emits events)
    DaemonType.QUEUE_POPULATOR,        # 6. Work queue maintenance
    DaemonType.WORK_QUEUE_MONITOR,     # 7. Queue visibility (after populator)
    DaemonType.COORDINATOR_HEALTH_MONITOR,  # 8. Coordinator visibility
    DaemonType.IDLE_RESOURCE,          # 9. GPU utilization
    DaemonType.TRAINING_TRIGGER,       # 10. Training trigger (after pipeline)

    # =========================================================================
    # Monitoring daemons (positions 11-16) - Dec 27, 2025
    # Added: 8 daemons missing from startup order per exploration analysis
    # Dec 30, 2025: Added MEMORY_MONITOR for 48-hour autonomous operation
    # =========================================================================
    DaemonType.CLUSTER_MONITOR,        # 11. Cluster monitoring (depends on EVENT_ROUTER)
    DaemonType.NODE_HEALTH_MONITOR,    # 12. Node health (depends on EVENT_ROUTER)
    DaemonType.HEALTH_SERVER,          # 13. Health endpoints (depends on EVENT_ROUTER)
    DaemonType.CLUSTER_WATCHDOG,       # 14. Cluster watchdog (depends on CLUSTER_MONITOR)
    DaemonType.NODE_RECOVERY,          # 15. Node recovery (depends on NODE_HEALTH_MONITOR)
    DaemonType.MEMORY_MONITOR,         # 16. Memory/VRAM monitor (prevents OOM crashes)
    DaemonType.SOCKET_LEAK_RECOVERY,   # 17. Socket/FD leak recovery (Jan 2026)

    # =========================================================================
    # Quality and training enhancement (positions 16-19) - Dec 27, 2025
    # Added: 3 daemons missing from startup order per exploration analysis
    # Dec 29, 2025: Added NNUE_TRAINING for automatic NNUE model training
    # =========================================================================
    DaemonType.QUALITY_MONITOR,        # 16. Quality monitoring (depends on DATA_PIPELINE)
    DaemonType.NNUE_TRAINING,          # 17. NNUE training (depends on DATA_PIPELINE)
    DaemonType.ARCHITECTURE_FEEDBACK,  # 18. Architecture allocation weights (Dec 29, 2025)
    DaemonType.DISTILLATION,           # 19. Distillation (depends on TRAINING_TRIGGER)

    # =========================================================================
    # Evaluation and promotion chain (positions 20-24)
    # Must be in order: EVALUATION -> (UNIFIED_PROMOTION) -> AUTO_PROMOTION -> MODEL_DISTRIBUTION
    # =========================================================================
    DaemonType.EVALUATION,             # 20. Model evaluation (depends on TRAINING_TRIGGER)
    DaemonType.UNIFIED_PROMOTION,      # 21. Unified promotion (depends on EVALUATION)
    DaemonType.AUTO_PROMOTION,         # 22. Auto-promotion (depends on EVALUATION)
    DaemonType.MODEL_DISTRIBUTION,     # 23. Model distribution (depends on AUTO_PROMOTION)
]


# P0.4 Dec 2025: Explicit dependency map for startup validation
# Daemons MUST wait for all dependencies to be running before starting.
# This prevents race conditions where events are emitted before handlers are ready.
DAEMON_DEPENDENCIES: dict[DaemonType, set[DaemonType]] = {
    # Core infrastructure (no dependencies)
    DaemonType.EVENT_ROUTER: set(),
    DaemonType.DAEMON_WATCHDOG: {DaemonType.EVENT_ROUTER},

    # Pipeline processors (depend on event system)
    DaemonType.DATA_PIPELINE: {DaemonType.EVENT_ROUTER},
    DaemonType.FEEDBACK_LOOP: {DaemonType.EVENT_ROUTER},

    # Sync daemons (depend on pipeline being ready to handle events)
    DaemonType.AUTO_SYNC: {
        DaemonType.EVENT_ROUTER,
        DaemonType.DATA_PIPELINE,
        DaemonType.FEEDBACK_LOOP,
    },

    # Queue management (depend on event system)
    DaemonType.QUEUE_POPULATOR: {DaemonType.EVENT_ROUTER},
    DaemonType.WORK_QUEUE_MONITOR: {DaemonType.EVENT_ROUTER, DaemonType.QUEUE_POPULATOR},
    DaemonType.COORDINATOR_HEALTH_MONITOR: {DaemonType.EVENT_ROUTER},

    # Resource management (depend on queue being populated)
    DaemonType.IDLE_RESOURCE: {DaemonType.EVENT_ROUTER, DaemonType.QUEUE_POPULATOR},

    # Training coordination (depend on pipeline and sync)
    DaemonType.TRAINING_TRIGGER: {
        DaemonType.EVENT_ROUTER,
        DaemonType.DATA_PIPELINE,
        DaemonType.AUTO_SYNC,
    },

    # Evaluation daemons
    DaemonType.EVALUATION: {DaemonType.EVENT_ROUTER, DaemonType.TRAINING_TRIGGER},
    DaemonType.AUTO_PROMOTION: {DaemonType.EVENT_ROUTER, DaemonType.EVALUATION},

    # Distribution daemons
    DaemonType.MODEL_DISTRIBUTION: {DaemonType.EVENT_ROUTER, DaemonType.AUTO_PROMOTION},
    DaemonType.NPZ_DISTRIBUTION: {DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE},

    # P2P daemons
    DaemonType.GOSSIP_SYNC: {DaemonType.EVENT_ROUTER},
    DaemonType.P2P_BACKEND: set(),  # Runs independently
    DaemonType.P2P_AUTO_DEPLOY: {DaemonType.EVENT_ROUTER},

    # Cluster consolidation (pulls games from cluster nodes to canonical DBs)
    DaemonType.CLUSTER_CONSOLIDATION: {
        DaemonType.EVENT_ROUTER,
        DaemonType.DATA_PIPELINE,
    },

    # Monitoring daemons
    DaemonType.CLUSTER_MONITOR: {DaemonType.EVENT_ROUTER},
    DaemonType.QUEUE_MONITOR: {DaemonType.EVENT_ROUTER},
    DaemonType.REPLICATION_MONITOR: {DaemonType.EVENT_ROUTER},

    # Multi-provider orchestrator
    DaemonType.MULTI_PROVIDER: {DaemonType.EVENT_ROUTER, DaemonType.IDLE_RESOURCE},

    # =========================================================================
    # Additional daemon dependencies (December 2025 - P0 CRITICAL fix)
    # =========================================================================

    # Sync daemons
    DaemonType.HIGH_QUALITY_SYNC: {DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE},
    DaemonType.ELO_SYNC: {DaemonType.EVENT_ROUTER},
    DaemonType.MODEL_SYNC: {DaemonType.EVENT_ROUTER},

    # Health/monitoring daemons
    DaemonType.NODE_HEALTH_MONITOR: {DaemonType.EVENT_ROUTER},
    DaemonType.SYSTEM_HEALTH_MONITOR: {DaemonType.EVENT_ROUTER},  # Deprecated but may still be used
    DaemonType.HEALTH_SERVER: {DaemonType.EVENT_ROUTER},
    DaemonType.CLUSTER_WATCHDOG: {DaemonType.EVENT_ROUTER, DaemonType.CLUSTER_MONITOR},
    DaemonType.MODEL_PERFORMANCE_WATCHDOG: {DaemonType.EVENT_ROUTER},

    # Event processing daemons
    DaemonType.CROSS_PROCESS_POLLER: {DaemonType.EVENT_ROUTER},
    DaemonType.DLQ_RETRY: {DaemonType.EVENT_ROUTER},

    # Pipeline/selfplay daemons
    DaemonType.SELFPLAY_COORDINATOR: {DaemonType.EVENT_ROUTER},
    DaemonType.CONTINUOUS_TRAINING_LOOP: {DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE},
    DaemonType.QUALITY_MONITOR: {DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE},

    # P2P/data server daemons
    DaemonType.DATA_SERVER: {DaemonType.EVENT_ROUTER},

    # Training enhancement daemons
    DaemonType.DISTILLATION: {DaemonType.EVENT_ROUTER, DaemonType.TRAINING_TRIGGER},
    DaemonType.UNIFIED_PROMOTION: {DaemonType.EVENT_ROUTER, DaemonType.EVALUATION},
    DaemonType.EXTERNAL_DRIVE_SYNC: {DaemonType.EVENT_ROUTER},
    DaemonType.VAST_CPU_PIPELINE: {DaemonType.EVENT_ROUTER},

    # Node watching/recovery daemons
    DaemonType.TRAINING_NODE_WATCHER: {DaemonType.EVENT_ROUTER},
    DaemonType.TRAINING_DATA_SYNC: {DaemonType.EVENT_ROUTER},  # Pre-training data sync
    DaemonType.TRAINING_WATCHDOG: {DaemonType.EVENT_ROUTER},  # Jan 4, 2026: Sprint 17
    DaemonType.EXPORT_WATCHDOG: {DaemonType.EVENT_ROUTER},  # Jan 6, 2026: Session 17.41
    # OWC import daemon (December 29, 2025) - imports from OWC external drive
    # December 30, 2025: Removed DATA_PIPELINE - not needed for file import
    DaemonType.OWC_IMPORT: {DaemonType.EVENT_ROUTER},
    # Production game import (January 2026) - imports human vs AI games from ringrift.ai
    DaemonType.PRODUCTION_GAME_IMPORT: {DaemonType.EVENT_ROUTER},
    DaemonType.NODE_RECOVERY: {DaemonType.EVENT_ROUTER, DaemonType.NODE_HEALTH_MONITOR},

    # Replication daemons
    DaemonType.REPLICATION_REPAIR: {DaemonType.EVENT_ROUTER, DaemonType.REPLICATION_MONITOR},

    # Tournament/evaluation daemons
    DaemonType.TOURNAMENT_DAEMON: {DaemonType.EVENT_ROUTER},
    DaemonType.GAUNTLET_FEEDBACK: {DaemonType.EVENT_ROUTER, DaemonType.EVALUATION},
    # Sprint 15: Backlog evaluation depends on EVENT_ROUTER for backpressure signals
    DaemonType.BACKLOG_EVALUATION: {DaemonType.EVENT_ROUTER},
    # Sprint 17.9: Comprehensive model scan discovers and queues models for multi-harness evaluation
    DaemonType.COMPREHENSIVE_MODEL_SCAN: {DaemonType.EVENT_ROUTER},

    # Data management daemons
    DaemonType.ORPHAN_DETECTION: {DaemonType.EVENT_ROUTER},
    DaemonType.AUTO_EXPORT: {DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE},
    DaemonType.DATA_CLEANUP: {DaemonType.EVENT_ROUTER, DaemonType.AUTO_SYNC},

    # Data consolidation depends on event router and data pipeline
    DaemonType.DATA_CONSOLIDATION: {DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE},
    # NPZ combination (Dec 2025) - combines NPZ files after export
    DaemonType.NPZ_COMBINATION: {DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE},
    # Comprehensive consolidation (Jan 2026) - scheduled sweep across all storage patterns
    DaemonType.COMPREHENSIVE_CONSOLIDATION: {DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE},
    DaemonType.S3_BACKUP: {DaemonType.EVENT_ROUTER, DaemonType.AUTO_PROMOTION},
    # S3 node sync (December 2025) - bi-directional S3 sync for cluster nodes
    DaemonType.S3_NODE_SYNC: {DaemonType.EVENT_ROUTER},
    # S3 consolidation (December 2025) - consolidates data from all nodes (coordinator only)
    DaemonType.S3_CONSOLIDATION: {DaemonType.EVENT_ROUTER, DaemonType.S3_NODE_SYNC},
    # Integrity check (December 2025) - scans for games without move data
    DaemonType.INTEGRITY_CHECK: {DaemonType.EVENT_ROUTER},

    # Job/resource daemons
    DaemonType.JOB_SCHEDULER: {DaemonType.EVENT_ROUTER},
    DaemonType.RESOURCE_OPTIMIZER: {DaemonType.EVENT_ROUTER},
    DaemonType.UTILIZATION_OPTIMIZER: {DaemonType.EVENT_ROUTER, DaemonType.IDLE_RESOURCE},

    # Curriculum/feedback daemons
    DaemonType.CURRICULUM_INTEGRATION: {DaemonType.EVENT_ROUTER, DaemonType.FEEDBACK_LOOP},

    # Recovery/coordination daemons
    DaemonType.RECOVERY_ORCHESTRATOR: {DaemonType.EVENT_ROUTER},
    DaemonType.CACHE_COORDINATION: {DaemonType.EVENT_ROUTER},
    DaemonType.METRICS_ANALYSIS: {DaemonType.EVENT_ROUTER},
    DaemonType.ADAPTIVE_RESOURCES: {DaemonType.EVENT_ROUTER, DaemonType.IDLE_RESOURCE},

    # Maintenance daemons
    DaemonType.MAINTENANCE: {DaemonType.EVENT_ROUTER},

    # Disk space management
    DaemonType.DISK_SPACE_MANAGER: {DaemonType.EVENT_ROUTER},
    # Coordinator disk manager (Dec 27, 2025) - specialized for coordinator-only nodes
    DaemonType.COORDINATOR_DISK_MANAGER: {DaemonType.EVENT_ROUTER},

    # Sync push daemon (Dec 28, 2025) - GPU nodes push data before cleanup
    DaemonType.SYNC_PUSH: {DaemonType.EVENT_ROUTER},

    # Provider-specific idle daemons
    DaemonType.VAST_IDLE: {DaemonType.EVENT_ROUTER},
    DaemonType.LAMBDA_IDLE: {DaemonType.EVENT_ROUTER},  # Deprecated but may still be referenced

    # Deprecated daemons (empty deps - should not be started)
    DaemonType.SYNC_COORDINATOR: set(),  # DEPRECATED: Use AUTO_SYNC
    DaemonType.HEALTH_CHECK: set(),  # DEPRECATED: Use NODE_HEALTH_MONITOR
    DaemonType.CLUSTER_DATA_SYNC: set(),  # DEPRECATED: Use AUTO_SYNC
    DaemonType.EPHEMERAL_SYNC: set(),  # DEPRECATED: Use AUTO_SYNC

    # =========================================================================
    # Cluster Availability Manager dependencies (December 28, 2025)
    # =========================================================================
    DaemonType.AVAILABILITY_NODE_MONITOR: {DaemonType.EVENT_ROUTER},
    DaemonType.AVAILABILITY_RECOVERY_ENGINE: {
        DaemonType.EVENT_ROUTER,
        DaemonType.AVAILABILITY_NODE_MONITOR,
    },
    DaemonType.AVAILABILITY_CAPACITY_PLANNER: {DaemonType.EVENT_ROUTER},
    DaemonType.AVAILABILITY_PROVISIONER: {
        DaemonType.EVENT_ROUTER,
        DaemonType.AVAILABILITY_CAPACITY_PLANNER,
    },

    # Tailscale health monitoring (December 29, 2025)
    # Runs independently on each node to monitor and auto-recover Tailscale
    DaemonType.TAILSCALE_HEALTH: set(),  # No dependencies - runs independently

    # Connectivity recovery coordinator (December 29, 2025)
    # Subscribes to TAILSCALE_*, HOST_*, P2P_NODE_DEAD events
    # Coordinates SSH-based recovery and escalation
    DaemonType.CONNECTIVITY_RECOVERY: {DaemonType.EVENT_ROUTER, DaemonType.TAILSCALE_HEALTH},

    # NNUE automatic training (December 29, 2025)
    # Trains NNUE models when game thresholds are met
    DaemonType.NNUE_TRAINING: {DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE},
    DaemonType.ARCHITECTURE_FEEDBACK: {DaemonType.EVENT_ROUTER},  # Dec 29, 2025

    # Parity validation daemon (December 30, 2025)
    # Runs on coordinator only - validates TS/Python parity and stores hashes
    DaemonType.PARITY_VALIDATION: {DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE},

    # =========================================================================
    # 48-Hour Autonomous Operation daemons (December 30, 2025)
    # =========================================================================
    DaemonType.PROGRESS_WATCHDOG: {DaemonType.EVENT_ROUTER},
    DaemonType.P2P_RECOVERY: {DaemonType.EVENT_ROUTER},
    DaemonType.VOTER_HEALTH_MONITOR: {DaemonType.EVENT_ROUTER},  # Dec 30, 2025: Continuous voter probing
    DaemonType.MEMORY_MONITOR: {DaemonType.EVENT_ROUTER},  # Emits MEMORY_PRESSURE events
    DaemonType.SOCKET_LEAK_RECOVERY: {DaemonType.EVENT_ROUTER},  # Emits SOCKET_LEAK events
    DaemonType.STALE_FALLBACK: {DaemonType.EVENT_ROUTER, DaemonType.AUTO_SYNC},  # Fallback when sync fails
    DaemonType.ELO_PROGRESS: {DaemonType.EVENT_ROUTER},  # Dec 31, 2025: Tracks Elo improvement
    DaemonType.PIPELINE_COMPLETENESS_MONITOR: {DaemonType.EVENT_ROUTER},  # Feb 2026: Pipeline stage tracking
}


def validate_daemon_dependencies(
    daemon_type: DaemonType,
    running_daemons: set[DaemonType],
) -> tuple[bool, list[DaemonType]]:
    """Check if all dependencies for a daemon are running.

    Args:
        daemon_type: The daemon to check dependencies for.
        running_daemons: Set of currently running daemon types.

    Returns:
        Tuple of (all_satisfied, missing_deps).
        If all_satisfied is True, missing_deps will be empty.
    """
    required = DAEMON_DEPENDENCIES.get(daemon_type, set())
    missing = [dep for dep in required if dep not in running_daemons]
    return (len(missing) == 0, missing)


def get_daemon_startup_position(daemon_type: DaemonType) -> int:
    """Get the startup position for a daemon in DAEMON_STARTUP_ORDER.

    Args:
        daemon_type: The daemon type to look up.

    Returns:
        Position (0-indexed) in startup order, or -1 if not in order.
        Daemons not in the order list can start after ordered daemons.
    """
    try:
        return DAEMON_STARTUP_ORDER.index(daemon_type)
    except ValueError:
        return -1


def validate_startup_order_consistency() -> tuple[bool, list[str]]:
    """Validate that DAEMON_STARTUP_ORDER is consistent with DAEMON_DEPENDENCIES.

    For each daemon in DAEMON_STARTUP_ORDER, if it depends on another daemon in
    the order, that dependency MUST come BEFORE it. Otherwise, the dependency
    won't be running when the daemon starts.

    Returns:
        Tuple of (is_consistent, violations).
        If is_consistent is True, violations will be empty.

    Example violations:
        - AUTO_SYNC at position 5 depends on DATA_PIPELINE at position 3 ✓ OK
        - DATA_PIPELINE at position 3 depends on AUTO_SYNC at position 5 ✗ VIOLATION

    December 2025: Added to prevent race conditions from startup order bugs.
    """
    violations: list[str] = []

    # Build position lookup for daemons in the startup order
    positions = {dt: i for i, dt in enumerate(DAEMON_STARTUP_ORDER)}

    for position, daemon_type in enumerate(DAEMON_STARTUP_ORDER):
        deps = DAEMON_DEPENDENCIES.get(daemon_type, set())

        for dep in deps:
            if dep in positions:
                dep_position = positions[dep]
                if dep_position > position:
                    # Dependency comes AFTER this daemon - violation!
                    violations.append(
                        f"{daemon_type.value} (pos {position}) depends on "
                        f"{dep.value} (pos {dep_position}), but {dep.value} "
                        f"starts AFTER {daemon_type.value}"
                    )

    return (len(violations) == 0, violations)


def validate_startup_order_or_raise() -> None:
    """Validate startup order consistency or raise an error.

    Raises:
        ValueError: If DAEMON_STARTUP_ORDER is inconsistent with DAEMON_DEPENDENCIES.
    """
    is_consistent, violations = validate_startup_order_consistency()
    if not is_consistent:
        error_msg = (
            "DAEMON_STARTUP_ORDER is inconsistent with DAEMON_DEPENDENCIES:\n"
            + "\n".join(f"  - {v}" for v in violations)
            + "\n\nFix: Either reorder DAEMON_STARTUP_ORDER or update DAEMON_DEPENDENCIES."
        )
        raise ValueError(error_msg)


# =============================================================================
# Callback Registration Pattern for Breaking Circular Dependencies (Dec 2025)
# =============================================================================
# Instead of importing daemon_manager, we use a callback that daemon_manager
# registers when it initializes. This breaks the daemon_types → daemon_manager
# circular dependency.

_mark_ready_callback: "Callable[[DaemonType], None] | None" = None


def register_mark_ready_callback(callback: "Callable[[DaemonType], None]") -> None:
    """Register the callback for mark_daemon_ready().

    This is called by DaemonManager.__init__() to provide the implementation
    without creating a circular import.

    Args:
        callback: Function that takes a DaemonType and signals readiness.
    """
    global _mark_ready_callback
    _mark_ready_callback = callback


def mark_daemon_ready(daemon_type: DaemonType) -> None:
    """Signal that a daemon has completed initialization and is ready.

    Daemons should call this after completing their initialization to unblock
    dependent daemons that are waiting to start.

    Usage in daemon factory:
        async def _create_my_daemon(self) -> None:
            # ... initialization code ...
            await some_initialization()

            # Signal readiness so dependent daemons can start
            mark_daemon_ready(DaemonType.MY_DAEMON)

            # ... main loop ...
            while True:
                await asyncio.sleep(60)

    Note:
        This function uses a callback pattern to avoid circular imports.
        The callback is registered by DaemonManager.__init__().
    """
    if _mark_ready_callback is not None:
        _mark_ready_callback(daemon_type)
    else:
        # Fallback: callback not registered yet (rare race condition at startup)
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(
            f"[DaemonTypes] mark_daemon_ready called for {daemon_type.value} "
            "before DaemonManager initialized"
        )
