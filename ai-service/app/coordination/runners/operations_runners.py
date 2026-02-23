"""Operations daemon runners: Feedback, Recovery, Miscellaneous, Data Integrity,
Cluster Availability, 48-Hour Autonomous, and Data Availability.

February 2026: Extracted from daemon_runners.py.

Contains runners for:
- Feedback & Curriculum Daemons
- Recovery & Maintenance Daemons
- Miscellaneous Daemons
- Data Integrity Daemons
- Cluster Availability Manager Daemons
- 48-Hour Autonomous Operation Daemons
- Data Availability Infrastructure Daemons
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Any

from app.coordination.runners import _wait_for_daemon

logger = logging.getLogger(__name__)


# =============================================================================
# Feedback & Curriculum Daemons
# =============================================================================


async def create_feedback_loop() -> None:
    """Create and run feedback loop controller (December 2025)."""
    try:
        from app.coordination.feedback_loop_controller import FeedbackLoopController

        controller = FeedbackLoopController()
        await controller.start()
        await _wait_for_daemon(controller)
    except ImportError as e:
        logger.error(f"FeedbackLoopController not available: {e}")
        raise


async def create_curriculum_integration() -> None:
    """Create and run curriculum integration daemon (December 2025)."""
    try:
        from app.coordination.curriculum_integration import (
            MomentumToCurriculumBridge,
        )

        bridge = MomentumToCurriculumBridge()
        # Dec 2025 fix: start() is synchronous (uses threading internally), don't await
        bridge.start()
        await _wait_for_daemon(bridge)
    except ImportError as e:
        logger.error(f"MomentumToCurriculumBridge not available: {e}")
        raise


# =============================================================================
# Recovery & Maintenance Daemons
# =============================================================================


async def create_recovery_orchestrator() -> None:
    """Create and run recovery orchestrator (December 2025)."""
    try:
        from app.coordination.recovery_orchestrator import RecoveryOrchestrator

        orchestrator = RecoveryOrchestrator()
        await orchestrator.start()
        await _wait_for_daemon(orchestrator)
    except ImportError as e:
        logger.error(f"RecoveryOrchestrator not available: {e}")
        raise


async def create_cache_coordination() -> None:
    """Create and run cache coordination orchestrator (December 2025).

    CacheCoordinationOrchestrator is event-driven. We instantiate it,
    call subscribe_to_events(), and keep it alive.
    """
    try:
        from app.coordination.cache_coordination_orchestrator import (
            CacheCoordinationOrchestrator,
        )

        orchestrator = CacheCoordinationOrchestrator()
        # CacheCoordinationOrchestrator needs explicit subscription call
        orchestrator.subscribe_to_events()
        logger.info("CacheCoordinationOrchestrator initialized and subscribed to events")

        # Keep alive by waiting indefinitely
        while True:
            await asyncio.sleep(60)
    except ImportError as e:
        logger.error(f"CacheCoordinationOrchestrator not available: {e}")
        raise


async def create_maintenance() -> None:
    """Create and run maintenance daemon (December 2025)."""
    try:
        from app.coordination.maintenance_daemon import MaintenanceDaemon

        daemon = MaintenanceDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"MaintenanceDaemon not available: {e}")
        raise


async def create_orphan_detection() -> None:
    """Create and run orphan detection daemon (December 2025)."""
    try:
        from app.coordination.orphan_detection_daemon import OrphanDetectionDaemon

        daemon = OrphanDetectionDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"OrphanDetectionDaemon not available: {e}")
        raise


async def create_data_cleanup() -> None:
    """Create and run data cleanup daemon (December 2025)."""
    try:
        from app.coordination.data_cleanup_daemon import DataCleanupDaemon

        daemon = DataCleanupDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"DataCleanupDaemon not available: {e}")
        raise


async def create_disk_space_manager() -> None:
    """Create and run disk space manager daemon (December 2025).

    Proactive disk space management:
    - Monitors disk usage across nodes
    - Triggers cleanup at 60% (before 70% warning)
    - Removes old logs, empty databases, old checkpoints
    - Emits DISK_SPACE_LOW and DISK_CLEANUP_TRIGGERED events
    """
    try:
        from app.coordination.disk_space_manager_daemon import DiskSpaceManagerDaemon

        daemon = DiskSpaceManagerDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"DiskSpaceManagerDaemon not available: {e}")
        raise


async def create_coordinator_disk_manager() -> None:
    """Create and run coordinator disk manager daemon (December 27, 2025).

    Specialized disk management for coordinator-only nodes:
    - Syncs data to remote storage (OWC on mac-studio) before cleanup
    - More aggressive cleanup thresholds (50% vs 60%)
    - Removes synced training/game files after 24 hours
    - Keeps canonical databases locally for quick access

    January 13, 2026: Added coordinator check to prevent running on GPU nodes.
    Only runs if RINGRIFT_IS_COORDINATOR=true environment variable is set.
    """
    import os

    is_coordinator = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() in (
        "true",
        "1",
        "yes",
    )
    if not is_coordinator:
        logger.info(
            "[CoordinatorDiskManager] Skipping - not a coordinator node "
            "(set RINGRIFT_IS_COORDINATOR=true to enable)"
        )
        return

    try:
        from app.coordination.disk_space_manager_daemon import CoordinatorDiskManager

        daemon = CoordinatorDiskManager()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"CoordinatorDiskManager not available: {e}")
        raise


async def create_node_availability() -> None:
    """Create and run node availability daemon (December 28, 2025).

    Synchronizes cloud provider instance state with distributed_hosts.yaml:
    - Queries Vast.ai, Lambda, RunPod APIs for current instance states
    - Updates YAML status when instances change (terminated, stopped, etc.)
    - Atomic YAML updates with backup
    - Dry-run mode by default for testing

    Solves the problem of stale config where nodes are marked 'ready'
    but are actually terminated in the cloud provider.
    """
    try:
        from app.coordination.node_availability.daemon import (
            NodeAvailabilityDaemon,
            get_node_availability_daemon,
        )

        daemon = get_node_availability_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"NodeAvailabilityDaemon not available: {e}")
        raise


async def create_sync_push() -> None:
    """Create and run sync push daemon (December 28, 2025).

    Push-based sync for GPU training nodes:
    - At 50% disk: Start pushing completed games to coordinator
    - At 70% disk: Push urgently (bypass write locks if safe)
    - At 75% disk: Clean up files with 2+ verified copies
    - Never delete files without verified sync receipts

    This is part of the "sync then clean" pattern to prevent data loss
    on GPU nodes that fill up during selfplay.
    """
    try:
        from app.coordination.sync_push_daemon import SyncPushDaemon

        daemon = SyncPushDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"SyncPushDaemon not available: {e}")
        raise


async def create_unified_data_plane() -> None:
    """Create and run unified data plane daemon (December 28, 2025).

    Consolidated data synchronization daemon that replaces fragmented sync
    infrastructure (~4,514 LOC consolidated):
    - AutoSyncDaemon (P2P gossip sync)
    - SyncFacade (backend routing)
    - S3NodeSyncDaemon (S3 backup)
    - dynamic_data_distribution.py (OWC distribution)
    - SyncRouter (intelligent routing)

    Key Components:
    - DataCatalog: Central registry of what data exists where
    - SyncPlanner v2: Intelligent routing (what to sync where and when)
    - TransportManager: Unified transfer layer with fallback chains
    - EventBridge: Event routing with chain completion

    Event Chain Completion:
        SELFPLAY_COMPLETE -> DATA_SYNC_STARTED -> DATA_SYNC_COMPLETED -> NEW_GAMES_AVAILABLE
        TRAINING_COMPLETED -> MODEL_PROMOTED -> MODEL_DISTRIBUTION_STARTED -> MODEL_DISTRIBUTION_COMPLETE
    """
    try:
        from app.coordination.unified_data_plane_daemon import (
            UnifiedDataPlaneDaemon,
            get_data_plane_daemon,
        )

        daemon = get_data_plane_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"UnifiedDataPlaneDaemon not available: {e}")
        raise


# =============================================================================
# Miscellaneous Daemons
# =============================================================================


async def create_s3_backup() -> None:
    """Create and run S3 backup daemon (December 2025)."""
    try:
        from app.coordination.s3_backup_daemon import S3BackupDaemon

        daemon = S3BackupDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"S3BackupDaemon not available: {e}")
        raise


async def create_s3_node_sync() -> None:
    """Create and run S3 node sync daemon (December 2025).

    Bi-directional S3 sync for all cluster nodes.
    """
    try:
        from app.coordination.s3_node_sync_daemon import S3NodeSyncDaemon

        daemon = S3NodeSyncDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"S3NodeSyncDaemon not available: {e}")
        raise


async def create_s3_consolidation() -> None:
    """Create and run S3 consolidation daemon (December 2025).

    Consolidates data from all nodes (coordinator only).
    """
    try:
        from app.coordination.s3_node_sync_daemon import S3ConsolidationDaemon

        daemon = S3ConsolidationDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"S3ConsolidationDaemon not available: {e}")
        raise


async def create_unified_backup() -> None:
    """Create and run unified backup daemon (January 2026).

    Backs up all selfplay games to OWC external drive and S3.
    Uses GameDiscovery to find databases from all storage patterns.
    """
    try:
        from app.coordination.unified_backup_daemon import UnifiedBackupDaemon

        daemon = UnifiedBackupDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"UnifiedBackupDaemon not available: {e}")
        raise


async def create_s3_push() -> None:
    """Create and run S3 push daemon (January 2026).

    Pushes all game databases, training NPZ files, and models to S3
    for backup and cluster-wide access.
    """
    try:
        from app.coordination.s3_push_daemon import S3PushDaemon

        daemon = S3PushDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"S3PushDaemon not available: {e}")
        raise


async def create_distillation() -> None:
    """Create and run distillation daemon.

    Jan 3, 2026: DEPRECATED - Standalone script at scripts/distillation_daemon.py.
    """
    logger.warning(
        "[DEPRECATED] DistillationDaemon is a standalone script. "
        "Use scripts/distillation_daemon.py directly. "
        "This daemon will be removed in Q2 2026."
    )
    # Don't raise - allow daemon manager to continue with other daemons
    return


async def create_external_drive_sync() -> None:
    """Create and run external drive sync daemon."""
    try:
        from app.coordination.external_drive_sync import ExternalDriveSyncDaemon

        daemon = ExternalDriveSyncDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ExternalDriveSyncDaemon not available: {e}")
        raise


async def create_vast_cpu_pipeline() -> None:
    """Create and run Vast CPU pipeline daemon."""
    try:
        from app.distributed.vast_cpu_pipeline import VastCpuPipelineDaemon

        daemon = VastCpuPipelineDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"VastCpuPipelineDaemon not available: {e}")
        raise


async def create_cluster_data_sync() -> None:
    """Create and run cluster data sync daemon.

    DEPRECATED (December 2025): Use AutoSyncDaemon with strategy="broadcast" instead.
    This runner is retained for backward compatibility and will be removed in Q2 2026.
    """
    warnings.warn(
        "DaemonType.CLUSTER_DATA_SYNC is deprecated. Use AutoSyncDaemon(strategy='broadcast') "
        "instead. Removal scheduled for Q2 2026.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from app.coordination.auto_sync_daemon import (
            AutoSyncConfig,
            AutoSyncDaemon,
            SyncStrategy,
        )

        config = AutoSyncConfig.from_config_file()
        config.strategy = SyncStrategy.BROADCAST
        daemon = AutoSyncDaemon(config=config)
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"AutoSyncDaemon not available for cluster sync: {e}")
        raise


async def create_p2p_backend() -> None:
    """Create and run P2P backend daemon (December 2025)."""
    try:
        from app.coordination.p2p_integration import P2PIntegration

        integration = P2PIntegration()
        await integration.start()
        await _wait_for_daemon(integration)
    except ImportError as e:
        logger.error(f"P2PIntegration not available: {e}")
        raise


async def create_p2p_auto_deploy() -> None:
    """Create and run P2P auto-deploy daemon (Phase 21.2)."""
    try:
        from app.coordination.p2p_auto_deployer import P2PAutoDeployer

        deployer = P2PAutoDeployer()
        await deployer.run_daemon()
        await _wait_for_daemon(deployer)
    except ImportError as e:
        logger.error(f"P2PAutoDeployer not available: {e}")
        raise


async def create_metrics_analysis() -> None:
    """Create and run metrics analysis orchestrator (December 2025)."""
    try:
        from app.coordination.metrics_analysis_orchestrator import (
            MetricsAnalysisOrchestrator,
        )

        orchestrator = MetricsAnalysisOrchestrator()
        await orchestrator.start()
        await _wait_for_daemon(orchestrator)
    except ImportError as e:
        logger.error(f"MetricsAnalysisOrchestrator not available: {e}")
        raise


async def create_per_orchestrator() -> None:
    """Create and run PER (Prioritized Experience Replay) orchestrator (December 2025).

    Monitors PER buffer events across the training system. Subscribes to
    PER_BUFFER_REBUILT and PER_PRIORITIES_UPDATED events.
    """
    import asyncio

    try:
        from app.training.per_orchestrator import wire_per_events

        # Wire events and get orchestrator
        orchestrator = wire_per_events()
        if orchestrator:
            logger.info("[PEROrchestrator] Started and subscribed to events")
            # Keep daemon running
            while True:
                await asyncio.sleep(60)
    except ImportError as e:
        logger.error(f"PEROrchestrator not available: {e}")
        raise


async def create_data_consolidation() -> None:
    """Create and run data consolidation daemon (December 2025).

    Consolidates scattered selfplay games into canonical databases.
    Subscribes to SELFPLAY_COMPLETE and NEW_GAMES_AVAILABLE events.
    Emits CONSOLIDATION_COMPLETE event after merging games.
    """
    try:
        from app.coordination.data_consolidation_daemon import get_consolidation_daemon

        daemon = get_consolidation_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"DataConsolidationDaemon not available: {e}")
        raise


async def create_cluster_consolidation() -> None:
    """Create and run cluster consolidation daemon (January 2026).

    Pulls selfplay games from cluster nodes into canonical databases.
    Critical for training pipeline: bridges distributed selfplay with training.

    Flow:
        1. Discovers alive cluster nodes via P2P
        2. Syncs selfplay.db from each node via Tailscale/SSH
        3. Merges games into canonical_*.db databases
        4. Emits CONSOLIDATION_COMPLETE event for training trigger

    Runs on coordinator only (configurable).
    """
    try:
        from app.coordination.cluster_consolidation_daemon import (
            get_cluster_consolidation_daemon,
        )

        daemon = get_cluster_consolidation_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ClusterConsolidationDaemon not available: {e}")
        raise


async def create_comprehensive_consolidation() -> None:
    """Create and run comprehensive consolidation daemon (January 2026).

    Scheduled sweep consolidation that finds ALL games across 14+ storage patterns.
    Unlike DATA_CONSOLIDATION (event-driven), this runs on a schedule to ensure
    no games are missed from:
    - owc_imports/ - Games imported from OWC external drive
    - synced/ - Games synced from P2P cluster
    - p2p_gpu/ - Games from GPU selfplay nodes
    - gumbel/ - High-quality Gumbel MCTS games
    - slurm/ - Games from SLURM cluster runs
    - And 9+ other patterns

    Flow:
        1. Runs on schedule (default: every 30 minutes)
        2. Uses GameDiscovery to scan all 14+ database patterns
        3. Merges all games into canonical_*.db databases
        4. Updates consolidation_tracking.db with progress
        5. Emits COMPREHENSIVE_CONSOLIDATION_COMPLETE event

    Sprint 1 of Unified Data Consolidation plan.
    """
    try:
        from app.coordination.comprehensive_consolidation_daemon import (
            get_comprehensive_consolidation_daemon,
        )

        daemon = get_comprehensive_consolidation_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ComprehensiveConsolidationDaemon not available: {e}")
        raise


async def create_unified_data_sync_orchestrator() -> None:
    """Create and run unified data sync orchestrator daemon (January 2026).

    Central coordinator for all data synchronization operations:
    - Listens to DATA_SYNC_COMPLETED events from AutoSyncDaemon
    - Triggers S3 and OWC backups based on event flags
    - Tracks replication status across all storage destinations
    - Provides unified visibility into data distribution

    Part of unified data sync plan for comprehensive backup and visibility.
    """
    try:
        from app.coordination.unified_data_sync_orchestrator import (
            get_unified_data_sync_orchestrator,
        )

        daemon = get_unified_data_sync_orchestrator()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"UnifiedDataSyncOrchestrator not available: {e}")
        raise


async def create_npz_combination() -> None:
    """Create and run NPZ combination daemon (December 2025).

    Combines multiple NPZ training files with quality-aware weighting.
    Subscribes to NPZ_EXPORT_COMPLETE event to trigger combination.
    Emits NPZ_COMBINATION_COMPLETE event after combining files.
    """
    try:
        from app.coordination.npz_combination_daemon import get_npz_combination_daemon

        daemon = get_npz_combination_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"NPZCombinationDaemon not available: {e}")
        raise


# =============================================================================
# Data Integrity (December 2025)
# =============================================================================


async def create_integrity_check() -> None:
    """Create and run integrity check daemon (December 2025).

    Periodically scans game databases for integrity issues, specifically:
    - Games without move data (orphan games)
    - Quarantines invalid games for review
    - Cleans up old quarantined games

    Part of Phase 6: Move Data Integrity Enforcement.
    """
    try:
        from app.coordination.integrity_check_daemon import (
            IntegrityCheckConfig,
            IntegrityCheckDaemon,
        )

        config = IntegrityCheckConfig.from_env()
        daemon = IntegrityCheckDaemon(config=config)
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"IntegrityCheckDaemon not available: {e}")
        raise


# =============================================================================
# Cluster Availability Manager (December 28, 2025)
# =============================================================================


async def create_availability_node_monitor() -> None:
    """Create and run availability NodeMonitor daemon.

    Multi-layer health checking for cluster nodes:
    - P2P heartbeat checks
    - SSH connectivity
    - GPU health monitoring
    - Provider API status

    Emits NODE_UNHEALTHY, NODE_RECOVERED events.
    """
    try:
        from app.coordination.availability.node_monitor import get_node_monitor

        daemon = get_node_monitor()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"NodeMonitor not available: {e}")
        raise


async def create_availability_recovery_engine() -> None:
    """Create and run availability RecoveryEngine daemon.

    Escalating recovery strategies:
    - RESTART_P2P
    - RESTART_TAILSCALE
    - REBOOT_INSTANCE
    - RECREATE_INSTANCE

    Subscribes to NODE_UNHEALTHY events.
    Emits RECOVERY_INITIATED, RECOVERY_COMPLETED events.
    """
    try:
        from app.coordination.availability.recovery_engine import get_recovery_engine

        daemon = get_recovery_engine()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"RecoveryEngine not available: {e}")
        raise


async def create_availability_capacity_planner() -> None:
    """Create and run availability CapacityPlanner daemon.

    Budget-aware capacity management:
    - Tracks hourly/daily spending
    - Provides scaling recommendations
    - Monitors cluster utilization

    Subscribes to NODE_PROVISIONED, NODE_TERMINATED events.
    Emits BUDGET_ALERT events.
    """
    try:
        from app.coordination.availability.capacity_planner import get_capacity_planner

        daemon = get_capacity_planner()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"CapacityPlanner not available: {e}")
        raise


async def create_cascade_training() -> None:
    """Create and run CascadeTrainingOrchestrator daemon.

    December 29, 2025: Multiplayer bootstrapping via cascade training.
    Orchestrates the 2p -> 3p -> 4p training cascade:
    - Monitors model quality/Elo for each board type
    - Triggers weight transfer when quality threshold met
    - Boosts selfplay priority for configs blocking cascade

    Subscribes to: TRAINING_COMPLETED, EVALUATION_COMPLETED, MODEL_PROMOTED, ELO_UPDATED
    Emits: CASCADE_TRANSFER_TRIGGERED, TRAINING_REQUESTED
    """
    try:
        from app.coordination.cascade_training import get_cascade_orchestrator

        daemon = get_cascade_orchestrator()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"CascadeTrainingOrchestrator not available: {e}")
        raise


async def create_availability_provisioner() -> None:
    """Create and run availability Provisioner daemon.

    Auto-provision new instances when capacity drops below thresholds.
    Respects budget constraints via CapacityPlanner integration.

    Subscribes to CAPACITY_LOW, NODE_FAILED_PERMANENTLY events.
    Emits NODE_PROVISIONED, PROVISION_FAILED events.
    """
    try:
        from app.coordination.availability.provisioner import get_provisioner

        daemon = get_provisioner()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"Provisioner not available: {e}")
        raise


# =============================================================================
# 48-Hour Autonomous Operation Daemons (December 29, 2025)
# =============================================================================


async def create_progress_watchdog() -> None:
    """Create and run ProgressWatchdog daemon.

    December 29, 2025: Part of 48-hour autonomous operation enablement.
    Monitors Elo velocity across all 12 configs and detects training stalls.

    - Tracks Elo velocity per config
    - Detects stalls (6+ hours without progress)
    - Triggers recovery via PROGRESS_STALL_DETECTED event

    Subscribes to: ELO_UPDATED
    Emits: PROGRESS_STALL_DETECTED, PROGRESS_RECOVERED
    """
    try:
        from app.coordination.progress_watchdog_daemon import get_progress_watchdog

        daemon = get_progress_watchdog()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ProgressWatchdog not available: {e}")
        raise


async def create_p2p_recovery() -> None:
    """Create and run P2PRecovery daemon.

    December 29, 2025: Part of 48-hour autonomous operation enablement.
    Monitors P2P cluster health and auto-restarts on partition/failure.

    - Checks P2P /status endpoint every 60s
    - Tracks consecutive failures
    - Restarts P2P after 3 consecutive failures
    - Respects 5-minute cooldown between restarts

    Emits: P2P_RESTART_TRIGGERED, P2P_HEALTH_RECOVERED
    """
    try:
        from app.coordination.p2p_recovery_daemon import get_p2p_recovery_daemon

        daemon = get_p2p_recovery_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"P2PRecovery not available: {e}")
        raise


async def create_voter_health_monitor() -> None:
    """Create and run VoterHealthMonitor daemon.

    December 30, 2025: Part of 48-hour autonomous operation enablement.
    Monitors individual voter health with multi-transport probing.

    Probing order:
    - P2P HTTP /health endpoint (5s timeout)
    - Tailscale ping (10s timeout)
    - SSH connectivity check (15s timeout)

    Key events:
    - VOTER_OFFLINE: Individual voter became unreachable (after 2 consecutive failures)
    - VOTER_ONLINE: Individual voter recovered
    - QUORUM_LOST: Online voters dropped below quorum threshold
    - QUORUM_RESTORED: Quorum regained
    - QUORUM_AT_RISK: Exactly at quorum threshold (marginal)

    Subscribes to: None (cycle-based, 30s interval)
    Emits: VOTER_OFFLINE, VOTER_ONLINE, QUORUM_LOST, QUORUM_RESTORED, QUORUM_AT_RISK
    """
    try:
        from app.coordination.voter_health_daemon import get_voter_health_daemon

        daemon = get_voter_health_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"VoterHealthMonitor not available: {e}")
        raise


async def create_memory_monitor() -> None:
    """Create and run MemoryMonitor daemon.

    December 30, 2025: Part of 48-hour autonomous operation enablement.
    Monitors GPU VRAM and process RSS to detect memory pressure before OOM.

    Key thresholds:
    - GPU VRAM: 75% warning, 85% critical -> Emit MEMORY_PRESSURE, pause spawning
    - System RAM: 80% warning, 90% critical -> Emit RESOURCE_CONSTRAINT
    - Process RSS: 32GB critical -> SIGTERM -> wait 60s -> SIGKILL

    Subscribes to: None (cycle-based)
    Emits: MEMORY_PRESSURE, RESOURCE_CONSTRAINT
    """
    try:
        from app.coordination.memory_monitor_daemon import get_memory_monitor

        daemon = get_memory_monitor()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"MemoryMonitor not available: {e}")
        raise


async def create_socket_leak_recovery() -> None:
    """Create and run SocketLeakRecovery daemon.

    January 2026: Part of 48-hour autonomous operation enablement.
    Monitors socket connections and file descriptors to detect and recover
    from resource leaks before they cause system instability.

    Key features:
    - Monitors TIME_WAIT, CLOSE_WAIT socket buildup
    - Monitors file descriptor exhaustion
    - Triggers connection pool cleanup when critical
    - Part of resource recovery infrastructure with MEMORY_MONITOR

    Subscribes to: RESOURCE_CONSTRAINT
    Emits: SOCKET_LEAK_DETECTED, SOCKET_LEAK_RECOVERED, P2P_CONNECTION_RESET_REQUESTED
    """
    try:
        from app.coordination.socket_leak_recovery_daemon import (
            get_socket_leak_recovery_daemon,
        )

        daemon = get_socket_leak_recovery_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"SocketLeakRecoveryDaemon not available: {e}")
        raise


async def create_stale_fallback() -> None:
    """Create and run StaleFallback controller.

    December 30, 2025: Part of 48-hour autonomous operation enablement.
    Enables graceful degradation when sync fails by using older model versions.

    Note: TrainingFallbackController is a utility controller, not a full daemon.
    This runner initializes it and subscribes to relevant events.

    Subscribes to: DATA_SYNC_FAILED, DATA_SYNC_COMPLETED
    Emits: STALE_TRAINING_FALLBACK (via controller.should_allow_training)
    """
    try:
        from app.coordination.stale_fallback import get_training_fallback_controller

        controller = get_training_fallback_controller()

        # Subscribe to sync events
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if router:
                router.subscribe(
                    "DATA_SYNC_FAILED",
                    lambda e: controller.record_sync_failure(e.get("config_key", "unknown")),
                    handler_name="stale_fallback_sync_failed",
                )
                router.subscribe(
                    "DATA_SYNC_COMPLETED",
                    lambda e: controller.record_sync_success(e.get("config_key", "unknown")),
                    handler_name="stale_fallback_sync_completed",
                )
                logger.info("[StaleFallback] Subscribed to sync events")
        except Exception as e:
            logger.debug(f"[StaleFallback] Could not subscribe to events: {e}")

        # Keep running (controller is stateful)
        while True:
            await asyncio.sleep(60)
    except ImportError as e:
        logger.error(f"StaleFallback not available: {e}")
        raise


async def create_underutilization_recovery() -> None:
    """Create and run UnderutilizationRecoveryHandler.

    January 4, 2026: Phase 3 of P2P Cluster Resilience plan.

    Handles CLUSTER_UNDERUTILIZED and WORK_QUEUE_EXHAUSTED events by injecting
    high-priority work items for underserved configurations.

    Subscribes to: CLUSTER_UNDERUTILIZED, WORK_QUEUE_EXHAUSTED
    Emits: UTILIZATION_RECOVERY_STARTED, UTILIZATION_RECOVERY_COMPLETED,
           UTILIZATION_RECOVERY_FAILED
    """
    try:
        from app.coordination.underutilization_recovery_handler import (
            get_underutilization_handler,
        )

        handler = get_underutilization_handler()
        await handler.start()
        await _wait_for_daemon(handler)
    except ImportError as e:
        logger.error(f"UnderutilizationRecoveryHandler not available: {e}")
        raise


async def create_fast_failure_detector() -> None:
    """Create and run FastFailureDetector.

    January 4, 2026: Phase 4 of P2P Cluster Resilience plan.

    Detects cluster-wide failures within 5-10 minutes using tiered escalation:
    - Tier 1 (5 min): Warning log
    - Tier 2 (10 min): Emit FAST_FAILURE_ALERT, boost selfplay 1.5x
    - Tier 3 (30 min): Emit FAST_FAILURE_RECOVERY, boost selfplay 2x

    Emits: FAST_FAILURE_ALERT, FAST_FAILURE_RECOVERY, FAST_FAILURE_RECOVERED
    """
    try:
        from app.coordination.fast_failure_detector import get_fast_failure_detector

        detector = get_fast_failure_detector()
        await detector.start()
        await _wait_for_daemon(detector)
    except ImportError as e:
        logger.error(f"FastFailureDetector not available: {e}")
        raise


async def create_tailscale_health() -> None:
    """Create and run TailscaleHealthDaemon.

    December 29, 2025: Monitors and auto-recovers Tailscale connectivity.

    Runs on each cluster node to ensure P2P mesh stability:
    - Checks Tailscale status every 30s (configurable)
    - Detects disconnected/expired states
    - Auto-recovers via tailscale up or tailscaled restart
    - Supports userspace networking for containers

    Emits: TAILSCALE_DISCONNECTED, TAILSCALE_RECOVERED, TAILSCALE_RECOVERY_FAILED
    """
    try:
        from app.coordination.tailscale_health_daemon import get_tailscale_health_daemon

        daemon = get_tailscale_health_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"TailscaleHealthDaemon not available: {e}")
        raise


async def create_connectivity_recovery() -> None:
    """Create and run ConnectivityRecoveryCoordinator.

    December 29, 2025: Unified event-driven connectivity recovery.

    Bridges TailscaleHealthDaemon, NodeAvailabilityDaemon, and P2P orchestrator:
    - Subscribes to TAILSCALE_DISCONNECTED, TAILSCALE_RECOVERED events
    - Subscribes to P2P_NODE_DEAD, HOST_OFFLINE, HOST_ONLINE events
    - Triggers SSH-based Tailscale recovery for masked failures
    - Escalates persistent failures with Slack alerts
    - Tracks connectivity state for all nodes

    Emits: TAILSCALE_RECOVERED (after SSH recovery), CONNECTIVITY_RECOVERY_ESCALATED
    """
    try:
        from app.coordination.connectivity_recovery_coordinator import (
            get_connectivity_recovery_coordinator,
        )

        daemon = get_connectivity_recovery_coordinator()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ConnectivityRecoveryCoordinator not available: {e}")
        raise


# =============================================================================
# Data Availability Infrastructure (Jan 3, 2026)
# =============================================================================


async def create_dual_backup() -> None:
    """Create and run dual backup daemon (S3 + OWC)."""
    try:
        from app.coordination.dual_backup_daemon import (
            create_dual_backup as _create,
        )

        await _create()
    except ImportError as e:
        logger.error(f"DualBackupDaemon not available: {e}")
        raise


async def create_owc_push() -> None:
    """Create and run OWC push daemon."""
    try:
        from app.coordination.owc_push_daemon import (
            create_owc_push as _create,
        )

        await _create()
    except ImportError as e:
        logger.error(f"OWCPushDaemon not available: {e}")
        raise


async def create_s3_import() -> None:
    """Create and run S3 import daemon."""
    try:
        from app.coordination.s3_import_daemon import (
            create_s3_import as _create,
        )

        await _create()
    except ImportError as e:
        logger.error(f"S3ImportDaemon not available: {e}")
        raise


async def create_unified_data_catalog() -> None:
    """Create and run unified data catalog daemon (placeholder)."""
    logger.warning(
        "UNIFIED_DATA_CATALOG daemon is not yet implemented - "
        "this is a placeholder for future unified data catalog API"
    )
    # Placeholder - daemon does nothing but prevents startup failure
    await asyncio.sleep(float("inf"))


async def create_node_data_agent() -> None:
    """Create and run node data agent daemon (placeholder)."""
    logger.warning(
        "NODE_DATA_AGENT daemon is not yet implemented - "
        "this is a placeholder for future per-node data agent"
    )
    # Placeholder - daemon does nothing but prevents startup failure
    await asyncio.sleep(float("inf"))


async def create_online_merge() -> None:
    """Create and run online model merge daemon.

    January 2026: Part of human game training pipeline.
    Validates shadow models that have received online learning updates
    and merges them into canonical models if they show improvement.
    """
    try:
        from app.coordination.online_merge_daemon import get_online_merge_daemon

        daemon = get_online_merge_daemon()
        await daemon.start()
        await daemon.wait_until_stopped()
    except ImportError as e:
        logger.error(f"OnlineMergeDaemon not available: {e}")
        raise


async def create_owc_sync_manager() -> None:
    """Create and run OWC external drive sync manager daemon.

    February 2026: Consolidated OWC sync daemon that replaces:
    - EXTERNAL_DRIVE_SYNC
    - OWC_PUSH
    - DUAL_BACKUP
    - UNIFIED_BACKUP
    """
    try:
        from app.coordination.external_drive_sync import get_external_drive_sync_daemon

        daemon = get_external_drive_sync_daemon()
        await daemon.start()
    except ImportError as e:
        logger.warning(f"OWC Sync Manager not available (likely not on coordinator): {e}")
        # Don't raise - OWC sync is optional for non-coordinator nodes
        await asyncio.sleep(float("inf"))


async def create_s3_sync() -> None:
    """Create and run unified S3 sync daemon.

    February 2026: Consolidated S3 sync daemon that replaces:
    - S3_BACKUP
    - S3_PUSH
    - S3_NODE_SYNC
    """
    try:
        from app.coordination.s3_sync_daemon import get_s3_sync_daemon

        daemon = get_s3_sync_daemon()
        await daemon.start()
        await daemon.wait_until_stopped()
    except ImportError as e:
        logger.warning(f"S3 Sync Daemon not available: {e}")
        await asyncio.sleep(float("inf"))


async def create_production_game_import() -> None:
    """Create and run production game import daemon.

    February 2026: Imports games from production ringrift.ai server
    into the training pipeline.
    """
    try:
        from app.coordination.production_game_import_daemon import (
            get_production_game_import_daemon,
        )

        daemon = get_production_game_import_daemon()
        await daemon.start()
        await daemon.wait_until_stopped()
    except ImportError as e:
        logger.warning(f"Production Game Import daemon not available: {e}")
        await asyncio.sleep(float("inf"))


async def create_reanalysis() -> None:
    """Create and run reanalysis daemon.

    Re-evaluates historical games with improved models.
    Subscribes to MODEL_PROMOTED, triggers reanalysis on Elo improvements.
    """
    try:
        from app.coordination.reanalysis_daemon import get_reanalysis_daemon

        daemon = await get_reanalysis_daemon()
        await daemon.start()
        await daemon.wait_until_stopped()
    except ImportError as e:
        logger.warning(f"Reanalysis daemon not available: {e}")
        await asyncio.sleep(float("inf"))


async def create_pipeline_completeness_monitor() -> None:
    """Create and run pipeline completeness monitor daemon.

    Tracks pipeline stage completion timestamps per config and emits
    PIPELINE_STAGE_OVERDUE events when stages exceed thresholds.
    February 2026.
    """
    try:
        from app.coordination.pipeline_completeness_monitor import (
            PipelineCompletenessMonitor,
        )

        daemon = PipelineCompletenessMonitor.get_instance()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.warning(f"PipelineCompletenessMonitor not available: {e}")
        await asyncio.sleep(float("inf"))
