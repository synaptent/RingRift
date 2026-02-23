"""Sync and Event Processing daemon runners.

February 2026: Extracted from daemon_runners.py.

Contains runners for:
- Sync Daemons (sync_coordinator through gossip_sync)
- Event Processing Daemons (event_router, cross_process_poller, dlq_retry)
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Any

from app.coordination.runners import _wait_for_daemon

logger = logging.getLogger(__name__)


# =============================================================================
# Sync Daemons
# =============================================================================


async def create_sync_coordinator() -> None:
    """Create and run sync coordinator daemon.

    DEPRECATED (December 2025): Use DaemonType.AUTO_SYNC instead.
    This runner is retained for backward compatibility and will be removed in Q2 2026.
    """
    warnings.warn(
        "DaemonType.SYNC_COORDINATOR is deprecated. Use DaemonType.AUTO_SYNC instead. "
        "Removal scheduled for Q2 2026.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from app.distributed.sync_coordinator import SyncCoordinator

        sync = SyncCoordinator.get_instance()
        await sync.start()
        await _wait_for_daemon(sync)
    except ImportError as e:
        logger.error(f"SyncCoordinator not available: {e}")
        raise


async def create_high_quality_sync() -> None:
    """Create and run high quality data sync watcher (Phase 2, December 2025)."""
    try:
        from app.coordination.training_freshness import HighQualityDataSyncWatcher

        watcher = HighQualityDataSyncWatcher()
        await watcher.start()
        await _wait_for_daemon(watcher)
    except ImportError as e:
        logger.error(f"HighQualityDataSyncWatcher not available: {e}")
        raise


async def create_elo_sync() -> None:
    """Create and run Elo sync manager (Phase 8, December 2025)."""
    try:
        from app.tournament.elo_sync_manager import EloSyncManager

        manager = EloSyncManager()
        await manager.initialize()
        # December 27, 2025: Fixed - was calling sync_loop() which doesn't exist
        # The start() method from SyncManagerBase runs the sync loop
        await manager.start()
    except ImportError as e:
        logger.error(f"EloSyncManager not available: {e}")
        raise


async def create_auto_sync() -> None:
    """Create and run automated P2P data sync daemon (December 2025)."""
    try:
        from app.coordination.auto_sync_daemon import AutoSyncDaemon

        daemon = AutoSyncDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"AutoSyncDaemon not available: {e}")
        raise


async def create_config_sync() -> None:
    """Create and run config sync daemon (January 2026).

    Automatically syncs distributed_hosts.yaml across cluster nodes.
    - Coordinator: Detects file changes via mtime polling, emits CONFIG_UPDATED
    - Workers: Subscribe to event, pull newer config via rsync
    """
    try:
        from app.coordination.config_sync_daemon import get_config_sync_daemon

        daemon = get_config_sync_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ConfigSyncDaemon not available: {e}")
        raise


async def create_config_validator() -> None:
    """Create and run config validator daemon (January 2026).

    Validates distributed_hosts.yaml against provider APIs (Lambda, Vast,
    RunPod, Tailscale) to detect configuration drift and stale entries.
    """
    try:
        from app.coordination.config_validator_daemon import ConfigValidatorDaemon

        daemon = ConfigValidatorDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ConfigValidatorDaemon not available: {e}")
        raise


async def create_training_node_watcher() -> None:
    """Create and run training activity daemon (December 2025).

    Monitors cluster for training activity and triggers priority sync
    before training starts. Uses TrainingActivityDaemon which:
    - Detects training via P2P status (running_jobs, processes)
    - Detects local training via process monitoring
    - Triggers priority sync when new training detected
    - Emits TRAINING_STARTED events for coordination
    - Graceful shutdown on SIGTERM with final sync
    """
    try:
        from app.coordination.training_activity_daemon import (
            TrainingActivityConfig,
            TrainingActivityDaemon,
        )

        config = TrainingActivityConfig.from_env()
        daemon = TrainingActivityDaemon(config=config)
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"TrainingActivityDaemon not available: {e}")
        raise


async def create_training_data_sync() -> None:
    """Create and run training data sync daemon (December 2025).

    Pre-training data synchronization from OWC drive and S3.
    Ensures training nodes have access to the best available training
    data before starting training jobs.

    Features:
    - Multi-source: OWC external drive, S3 bucket
    - Config-aware: Only syncs data for configs being trained
    - Size-based selection: Downloads largest available dataset
    - Resume support: Skips already-downloaded files
    """
    try:
        from app.coordination.training_data_sync_daemon import (
            TrainingDataSyncDaemon,
            get_training_data_sync_daemon,
        )

        daemon = get_training_data_sync_daemon()
        await daemon.start()

        # Keep running until stopped
        while True:
            await asyncio.sleep(60)
            if not daemon._running:
                break

    except ImportError as e:
        logger.error(f"TrainingDataSyncDaemon not available: {e}")
        raise


async def create_training_data_recovery() -> None:
    """Create and run training data recovery daemon (January 2026 Sprint 13.3).

    Auto-recovery from training data corruption by re-exporting NPZ files.

    Features:
    - Subscribes to TRAINING_FAILED events
    - Detects data corruption patterns
    - Triggers NPZ re-export from canonical databases
    - Emits TRAINING_DATA_RECOVERED / TRAINING_DATA_RECOVERY_FAILED
    """
    try:
        from app.coordination.training_data_recovery_daemon import (
            TrainingDataRecoveryDaemon,
            get_training_data_recovery_daemon,
        )

        daemon = get_training_data_recovery_daemon()
        await daemon.start()

        # Keep running until stopped
        while True:
            await asyncio.sleep(60)
            if not daemon._running:
                break

    except ImportError as e:
        logger.error(f"TrainingDataRecoveryDaemon not available: {e}")
        raise


async def create_training_watchdog() -> None:
    """Create and run training watchdog daemon (January 2026 Sprint 17).

    Monitors training processes for stalls and kills stuck processes.

    Features:
    - Tracks training process heartbeats via TRAINING_HEARTBEAT events
    - Detects stale processes (no heartbeat for 2+ hours)
    - Kills stale processes (SIGTERM, then SIGKILL after grace period)
    - Releases associated training locks
    - Emits TRAINING_PROCESS_KILLED event
    """
    try:
        from app.coordination.training_watchdog_daemon import (
            TrainingWatchdogDaemon,
            get_training_watchdog_daemon,
        )

        daemon = get_training_watchdog_daemon()
        await daemon.start()

        # Keep running until stopped
        while True:
            await asyncio.sleep(60)
            if not daemon._running:
                break

    except ImportError as e:
        logger.error(f"TrainingWatchdogDaemon not available: {e}")
        raise


async def create_export_watchdog() -> None:
    """Create and run export watchdog daemon (January 2026 Session 17.41).

    Monitors export_replay_dataset.py processes for hangs and kills them
    if they exceed the maximum runtime threshold (default 30 minutes).

    Features:
    - Detects export processes via pgrep
    - Tracks runtime of each process
    - Kills processes exceeding max_export_runtime (SIGTERM, then SIGKILL)
    - Emits EXPORT_TIMEOUT event when killing a stuck process
    """
    try:
        from app.coordination.export_watchdog_daemon import (
            ExportWatchdogDaemon,
            get_export_watchdog_daemon,
        )

        daemon = get_export_watchdog_daemon()
        await daemon.start()

        # Keep running until stopped
        while True:
            await asyncio.sleep(60)
            if not daemon._running:
                break

    except ImportError as e:
        logger.error(f"ExportWatchdogDaemon not available: {e}")
        raise


async def create_owc_import() -> None:
    """Create and run OWC import daemon (December 29, 2025).

    Periodically imports training data from OWC external drive on mac-studio
    for underserved configs (those with <100,000 games in canonical databases).
    January 2026: Threshold increased from 500 to 100,000 to enable comprehensive
    import from OWC drive (8.5M games available).

    Features:
    - SSH-based discovery of databases on OWC drive
    - Automatic detection of underserved configs
    - Rsync-based database transfer
    - Event emission for consolidation pipeline integration
    - Emits NEW_GAMES_AVAILABLE and DATA_SYNC_COMPLETED events

    Environment variables:
    - OWC_HOST: Remote host with OWC drive (default: mac-studio)
    - OWC_BASE_PATH: Base path on OWC drive (default: /Volumes/RingRift-Data)
    - OWC_SSH_KEY: SSH key for connection (default: ~/.ssh/id_ed25519)
    - RINGRIFT_OWC_UNDERSERVED_THRESHOLD: Threshold for import (default: 100000)
    """
    try:
        from app.coordination.owc_import_daemon import (
            OWCImportConfig,
            OWCImportDaemon,
            get_owc_import_daemon,
        )

        daemon = get_owc_import_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"OWCImportDaemon not available: {e}")
        raise


async def create_owc_model_import() -> None:
    """Create and run OWC model import daemon (January 3, 2026 - Sprint 13 Session 4).

    Imports MODEL FILES (.pth) from OWC external drive on mac-studio.
    Unlike OWCImportDaemon (which imports databases), this daemon imports
    trained models that need Elo evaluation.

    Features:
    - SSH-based discovery of models on OWC drive
    - Cross-reference with EloService to skip already-rated models
    - Rsync-based model transfer
    - Emits MODEL_IMPORTED and OWC_MODELS_DISCOVERED events
    - Adds models to PersistentEvaluationQueue for evaluation

    Environment variables:
    - OWC_HOST: Remote host with OWC drive (default: mac-studio)
    - OWC_BASE_PATH: Base path on OWC drive (default: /Volumes/RingRift-Data)
    """
    try:
        from app.coordination.owc_model_import_daemon import (
            get_owc_model_import_daemon,
        )

        daemon = get_owc_model_import_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"OWCModelImportDaemon not available: {e}")
        raise


async def create_unevaluated_model_scanner() -> None:
    """Create and run unevaluated model scanner daemon (January 3, 2026 - Sprint 13 Session 4).

    Scans all model sources (local, OWC, cluster, registry) for models
    without Elo ratings. Adds them to the evaluation queue with
    curriculum-aware priority.

    Features:
    - Scans local models/ directory for .pth files
    - Cross-references with EloService (unified_elo.db)
    - Computes priority based on curriculum weights
    - Emits EVALUATION_REQUESTED and UNEVALUATED_MODELS_FOUND events
    - Adds models to PersistentEvaluationQueue
    """
    try:
        from app.coordination.unevaluated_model_scanner_daemon import (
            get_unevaluated_model_scanner_daemon,
        )

        daemon = get_unevaluated_model_scanner_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"UnevaluatedModelScannerDaemon not available: {e}")
        raise


async def create_stale_evaluation() -> None:
    """Create and run stale evaluation daemon (January 4, 2026 - Sprint 17).

    Re-evaluates models with Elo ratings older than a configured threshold
    (default 30 days). This ensures model rankings stay accurate as the
    training landscape evolves.

    Features:
    - Scans EloService for models with old evaluation timestamps
    - Adds to evaluation queue with appropriate priority
    - Prevents unnecessary re-evaluation with configurable cooldowns
    - Emits EVALUATION_REQUESTED events
    """
    try:
        from app.coordination.stale_evaluation_daemon import (
            get_stale_evaluation_daemon,
        )

        daemon = get_stale_evaluation_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"StaleEvaluationDaemon not available: {e}")
        raise


async def create_comprehensive_model_scan() -> None:
    """Create and run comprehensive model scan daemon (January 9, 2026 - Sprint 17.9).

    Discovers ALL models across all sources:
    - Local models/ directory
    - ClusterModelDiscovery (SSH queries to cluster nodes)
    - OWCModelDiscovery (external OWC drive)

    For each discovered model, queues evaluation under all compatible harnesses
    (BRS, MaxN, Descent, Gumbel MCTS). Ensures all models get fresh Elo ratings
    under multiple harnesses, with games saved for training.

    Features:
    - SHA256-based deduplication across sources
    - Priority-based queueing (4-player, underserved configs boosted)
    - Configurable harness types via environment
    - Subscribes to MODEL_IMPORTED and MODEL_PROMOTED for immediate queueing
    """
    try:
        from app.coordination.comprehensive_model_scan_daemon import (
            get_comprehensive_model_scan_daemon,
        )

        daemon = get_comprehensive_model_scan_daemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ComprehensiveModelScanDaemon not available: {e}")
        raise


async def create_ephemeral_sync() -> None:
    """Create and run ephemeral sync daemon (Phase 4, December 2025).

    DEPRECATED (December 2025): Use AutoSyncDaemon with strategy="ephemeral" instead.
    This runner is retained for backward compatibility and will be removed in Q2 2026.
    """
    warnings.warn(
        "DaemonType.EPHEMERAL_SYNC is deprecated. Use AutoSyncDaemon(strategy='ephemeral') instead. "
        "Removal scheduled for Q2 2026.",
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
        config.strategy = SyncStrategy.EPHEMERAL
        daemon = AutoSyncDaemon(config=config)
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"AutoSyncDaemon not available for ephemeral sync: {e}")
        raise


async def create_gossip_sync() -> None:
    """Create and run gossip sync daemon.

    Jan 3, 2026: Fixed to provide required constructor arguments.
    GossipSyncDaemon requires node_id, data_dir, and peers_config.

    Jan 10, 2026: Skip if P2P orchestrator is running (uses same port 8771).
    Feb 22, 2026: Always skip on coordinator - P2P orchestrator handles gossip.
    Binding 8771 here prevents P2P from starting if master_loop launches first.
    """
    import asyncio
    import socket

    from app.config.env import env

    # Coordinator nodes always use P2P orchestrator for gossip (port 8771).
    # If master_loop grabs 8771 first, P2P can't start - causing the entire
    # cluster pipeline to stall.
    if env.is_coordinator:
        logger.info("GossipSync skipped: coordinator uses P2P orchestrator for gossip")
        while True:
            await asyncio.sleep(3600)
        return

    # Check if port 8771 is already in use (likely P2P orchestrator)
    test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        test_sock.bind(("0.0.0.0", 8771))
        test_sock.close()
    except OSError:
        # Port already in use - P2P orchestrator likely running
        logger.info("GossipSync skipped: port 8771 already in use (P2P orchestrator running)")
        while True:
            await asyncio.sleep(3600)
        return

    try:
        from pathlib import Path

        from app.distributed.gossip_sync import GossipSyncDaemon

        # Get node_id from hostname
        node_id = socket.gethostname()

        # Use standard data directory
        data_dir = Path("data/games")

        # Load peers config from distributed_hosts.yaml
        peers_config: dict[str, dict] = {}
        try:
            from app.config.cluster_config import get_config_cache
            config = get_config_cache().get_config()
            if config and "hosts" in config:
                for host_name, host_info in config["hosts"].items():
                    if isinstance(host_info, dict):
                        peers_config[host_name] = {
                            "host": host_info.get("tailscale_ip") or host_info.get("ssh_host", ""),
                            "port": host_info.get("gossip_port", 8771),
                        }
        except Exception as e:
            logger.warning(f"Failed to load peers config for GossipSync: {e}")

        daemon = GossipSyncDaemon(
            node_id=node_id,
            data_dir=data_dir,
            peers_config=peers_config,
        )
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"GossipSyncDaemon not available: {e}")
        raise


# =============================================================================
# Event Processing Daemons
# =============================================================================


async def create_event_router() -> None:
    """Create and run unified event router daemon.

    December 27, 2025: Added validation guards to catch import timing issues.
    The .start() method may not be available if circular imports leave
    the UnifiedEventRouter class incomplete during initialization.
    """
    try:
        from app.coordination.event_router import UnifiedEventRouter, get_router

        # Validate class has required method (guards against incomplete import)
        if not hasattr(UnifiedEventRouter, "start"):
            logger.error(
                "UnifiedEventRouter.start() not found - possible circular import issue"
            )
            raise RuntimeError("UnifiedEventRouter missing start() method")

        router = get_router()

        # Double-check instance has method (guards against class mismatch)
        if not hasattr(router, "start"):
            logger.error("Router instance missing start() - possible stale module cache")
            raise RuntimeError("Router instance missing start() method")

        await router.start()
        await _wait_for_daemon(router)
    except ImportError as e:
        logger.error(f"UnifiedEventRouter not available: {e}")
        raise


async def create_cross_process_poller() -> None:
    """Create and run cross-process event poller daemon.

    Jan 3, 2026: Fixed import to use cross_process_events module.
    Jan 3, 2026: Fixed to provide required process_name argument.
    Jan 4, 2026: Fixed await issue - start() is synchronous, not async.
    """
    try:
        from app.coordination.cross_process_events import CrossProcessEventPoller

        # process_name identifies this poller instance for event routing
        poller = CrossProcessEventPoller(
            process_name="daemon_manager",
            event_types=None,  # Subscribe to all event types
        )
        # start() is synchronous - spawns background polling thread
        poller.start()
        # _wait_for_daemon expects async start, but we're already started
        # Just verify the poller is running
        if hasattr(poller, "_running") and poller._running:
            logger.info("CrossProcessEventPoller started successfully")
    except ImportError as e:
        logger.error(f"CrossProcessEventPoller not available: {e}")
        raise


async def create_dlq_retry() -> None:
    """Create and run dead-letter queue retry daemon (December 2025)."""
    try:
        from app.coordination.dead_letter_queue import DLQRetryDaemon

        retry = DLQRetryDaemon()  # Uses get_dead_letter_queue() internally
        await retry.start()
        await _wait_for_daemon(retry)
    except ImportError as e:
        logger.error(f"DLQRetryDaemon not available: {e}")
        raise
