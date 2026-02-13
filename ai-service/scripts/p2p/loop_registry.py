"""Loop Registry - Extracted from P2POrchestrator._register_extracted_loops().

January 2026: Extracted ~1,580 LOC from p2p_orchestrator.py to reduce file size.

This module handles registration of all extracted loops with the LoopManager.
Each loop is configured with appropriate callbacks that delegate to the orchestrator.

Usage:
    from scripts.p2p.loop_registry import register_all_loops, LoopRegistrationResult

    result = register_all_loops(orchestrator, manager)
    if result.success:
        print(f"Registered {result.loops_registered} loops")
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import aiohttp
from aiohttp import ClientTimeout

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator
    from scripts.p2p.loops.base import LoopManager

logger = logging.getLogger(__name__)

# Default P2P port
DEFAULT_PORT = 8770


@dataclass
class LoopRegistrationResult:
    """Result of loop registration."""

    success: bool
    loops_registered: int = 0
    loops_failed: list[str] = field(default_factory=list)
    error: str | None = None


def _get_client_session(timeout: ClientTimeout | None = None) -> aiohttp.ClientSession:
    """Get an aiohttp client session with optional timeout."""
    if timeout is None:
        timeout = ClientTimeout(total=30)
    return aiohttp.ClientSession(timeout=timeout)


def register_all_loops(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
) -> LoopRegistrationResult:
    """Register all extracted loops with the LoopManager.

    Args:
        orchestrator: The P2POrchestrator instance providing callbacks.
        manager: The LoopManager to register loops with.

    Returns:
        LoopRegistrationResult with success status and count.
    """
    loops_registered = 0
    loops_failed: list[str] = []

    try:
        # Import loop classes
        from scripts.p2p.loops import (
            QueuePopulatorLoop,
            EloSyncLoop,
            JobReaperLoop,
            JobReassignmentLoop,
            IdleDetectionLoop,
            AutoScalingLoop,
            SpawnVerificationLoop,
            PredictiveScalingLoop,
            WorkQueueMaintenanceLoop,
            NATManagementLoop,
            ManifestCollectionLoop,
            ValidationLoop,
            ModelSyncLoop,
            DataManagementLoop,
            HttpServerHealthLoop,
            OrchestratorContext,
        )

        # Create unified context for loop callbacks
        ctx = OrchestratorContext.from_orchestrator(orchestrator)

        # Register each loop type
        loops_registered += _register_queue_populator(orchestrator, manager, ctx, loops_failed)
        loops_registered += _register_elo_sync(orchestrator, manager, ctx, loops_failed)
        loops_registered += _register_job_reaper(orchestrator, manager, ctx, loops_failed)
        loops_registered += _register_orphan_detection(orchestrator, manager, loops_failed)
        loops_registered += _register_idle_detection(orchestrator, manager, ctx, loops_failed)
        loops_registered += _register_spawn_verification(orchestrator, manager, ctx, loops_failed)
        loops_registered += _register_predictive_scaling(orchestrator, manager, ctx, loops_failed)
        loops_registered += _register_job_reassignment(orchestrator, manager, ctx, loops_failed)
        loops_registered += _register_auto_scaling(orchestrator, manager, loops_failed)
        loops_registered += _register_work_queue_maintenance(orchestrator, manager, ctx, loops_failed)
        loops_registered += _register_nat_management(orchestrator, manager, loops_failed)
        loops_registered += _register_manifest_collection(orchestrator, manager, ctx, loops_failed)
        loops_registered += _register_data_management(orchestrator, manager, ctx, loops_failed)
        loops_registered += _register_model_sync(orchestrator, manager, loops_failed)
        loops_registered += _register_model_fetch(orchestrator, manager, ctx, loops_failed)
        loops_registered += _register_validation(orchestrator, manager, ctx, loops_failed)
        loops_registered += _register_tailscale_peer_discovery(orchestrator, manager, ctx, loops_failed)
        loops_registered += _register_peer_cleanup(orchestrator, manager, loops_failed)
        loops_registered += _register_gossip_state_cleanup(orchestrator, manager, loops_failed)
        loops_registered += _register_gossip_peer_promotion(orchestrator, manager, loops_failed)
        loops_registered += _register_quorum_crisis_discovery(orchestrator, manager, loops_failed)
        loops_registered += _register_worker_pull(orchestrator, manager, ctx, loops_failed)
        loops_registered += _register_follower_discovery(orchestrator, manager, ctx, loops_failed)
        loops_registered += _register_self_healing(orchestrator, manager, ctx, loops_failed)
        loops_registered += _register_remote_p2p_recovery(orchestrator, manager, ctx, loops_failed)
        loops_registered += _register_leader_probe(orchestrator, manager, loops_failed)
        loops_registered += _register_leader_maintenance(orchestrator, manager, loops_failed)
        loops_registered += _register_relay_health(orchestrator, manager, loops_failed)
        loops_registered += _register_predictive_monitoring(orchestrator, manager, loops_failed)
        loops_registered += _register_training_sync(orchestrator, manager, ctx, loops_failed)
        loops_registered += _register_data_aggregation(orchestrator, manager, loops_failed)
        loops_registered += _register_health_aggregation(orchestrator, manager, loops_failed)
        loops_registered += _register_ip_discovery(orchestrator, manager, loops_failed)
        loops_registered += _register_tailscale_recovery(orchestrator, manager, loops_failed)
        loops_registered += _register_tailscale_keepalive(orchestrator, manager, loops_failed)
        loops_registered += _register_cluster_healing(orchestrator, manager, loops_failed)
        loops_registered += _register_udp_discovery(orchestrator, manager, loops_failed)
        loops_registered += _register_split_brain_detection(orchestrator, manager, loops_failed)
        loops_registered += _register_autonomous_queue_population(orchestrator, manager, loops_failed)
        loops_registered += _register_http_server_health(orchestrator, manager, loops_failed)
        loops_registered += _register_comprehensive_evaluation(orchestrator, manager, loops_failed)
        loops_registered += _register_evaluation_worker(orchestrator, manager, loops_failed)
        loops_registered += _register_tournament_data_pipeline(orchestrator, manager, loops_failed)
        loops_registered += _register_circuit_breaker_decay(orchestrator, manager, loops_failed)
        loops_registered += _register_voter_config_sync(orchestrator, manager, loops_failed)
        loops_registered += _register_stability_controller(orchestrator, manager, loops_failed)
        loops_registered += _register_peer_recovery(orchestrator, manager, loops_failed)
        loops_registered += _register_git_update(orchestrator, manager, loops_failed)
        loops_registered += _register_voter_heartbeat(orchestrator, manager, loops_failed)
        loops_registered += _register_reconnect_dead_peers(orchestrator, manager, loops_failed)
        loops_registered += _register_swim_membership(orchestrator, manager, loops_failed)

        logger.info(f"LoopRegistry: registered {loops_registered} loops")
        if loops_failed:
            logger.debug(f"LoopRegistry: {len(loops_failed)} loops not available: {loops_failed}")

        return LoopRegistrationResult(
            success=True,
            loops_registered=loops_registered,
            loops_failed=loops_failed,
        )

    except Exception as e:
        logger.error(f"LoopRegistry: failed to register loops: {e}")
        return LoopRegistrationResult(
            success=False,
            loops_registered=loops_registered,
            loops_failed=loops_failed,
            error=str(e),
        )


# =============================================================================
# Individual loop registration functions
# =============================================================================


def _register_queue_populator(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    ctx: Any,
    failed: list[str],
) -> int:
    """Register QueuePopulatorLoop."""
    try:
        from scripts.p2p.loops import QueuePopulatorLoop
        from scripts.p2p_orchestrator import NodeRole

        queue_populator = QueuePopulatorLoop(
            get_role=lambda: NodeRole.LEADER if orchestrator._is_leader() else NodeRole.FOLLOWER,
            get_selfplay_scheduler=lambda: orchestrator.selfplay_scheduler,
            notifier=orchestrator.notifier,
        )
        manager.register(queue_populator)
        orchestrator._queue_populator_loop = queue_populator
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"QueuePopulatorLoop: not available: {e}")
        failed.append("QueuePopulatorLoop")
        return 0


def _register_elo_sync(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    ctx: Any,
    failed: list[str],
) -> int:
    """Register EloSyncLoop."""
    try:
        from scripts.p2p.loops import EloSyncLoop

        if ctx.elo_sync_manager is not None:
            elo_sync = EloSyncLoop(
                get_elo_sync_manager=lambda: ctx.elo_sync_manager,
                get_sync_in_progress=ctx.sync_in_progress or (lambda: False),
            )
            manager.register(elo_sync)
            return 1
        return 0
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"EloSyncLoop: not available: {e}")
        failed.append("EloSyncLoop")
        return 0


def _register_job_reaper(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    ctx: Any,
    failed: list[str],
) -> int:
    """Register JobReaperLoop."""
    try:
        from scripts.p2p.loops import JobReaperLoop

        job_reaper = JobReaperLoop(
            get_active_jobs=ctx.get_active_jobs,
            cancel_job=ctx.cancel_job,
            get_job_heartbeats=ctx.get_job_heartbeats,
        )
        manager.register(job_reaper)
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"JobReaperLoop: not available: {e}")
        failed.append("JobReaperLoop")
        return 0


def _register_orphan_detection(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register OrphanProcessDetectionLoop."""
    try:
        from scripts.p2p.loops.job_loops import OrphanProcessDetectionLoop

        def _get_tracked_pids() -> set[int]:
            """Get set of PIDs currently tracked by job manager."""
            tracked_pids: set[int] = set()
            with orchestrator.jobs_lock:
                for job_type, jobs in orchestrator.active_jobs.items():
                    for job_id, job_info in jobs.items():
                        pid = None
                        if isinstance(job_info, dict):
                            pid = job_info.get("pid")
                        else:
                            pid = getattr(job_info, "pid", None)
                        if pid and isinstance(pid, int):
                            tracked_pids.add(pid)
            return tracked_pids

        orphan_detection = OrphanProcessDetectionLoop(
            get_tracked_pids=_get_tracked_pids,
        )
        manager.register(orphan_detection)
        logger.info("[LoopRegistry] OrphanProcessDetectionLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"OrphanProcessDetectionLoop: not available: {e}")
        failed.append("OrphanProcessDetectionLoop")
        return 0


def _register_idle_detection(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    ctx: Any,
    failed: list[str],
) -> int:
    """Register IdleDetectionLoop."""
    try:
        from scripts.p2p.loops import IdleDetectionLoop

        idle_detection = IdleDetectionLoop(
            get_role=ctx.get_role,
            get_peers=ctx.get_peers,
            get_work_queue=ctx.get_work_queue,
            on_idle_detected=ctx.auto_start_selfplay,
            on_zombie_detected=ctx.handle_zombie_detected,
        )
        manager.register(idle_detection)
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"IdleDetectionLoop: not available: {e}")
        failed.append("IdleDetectionLoop")
        return 0


def _register_spawn_verification(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    ctx: Any,
    failed: list[str],
) -> int:
    """Register SpawnVerificationLoop."""
    try:
        from scripts.p2p.loops import SpawnVerificationLoop

        if ctx.selfplay_scheduler is not None:
            spawn_verification = SpawnVerificationLoop(
                verify_pending_spawns=ctx.selfplay_scheduler.verify_pending_spawns,
                get_spawn_stats=lambda: {
                    "per_node": {
                        node_id: ctx.selfplay_scheduler.get_spawn_success_rate(node_id)
                        for node_id in list(ctx.get_peers().keys())[:20]
                    } if ctx.get_peers() else {},
                },
            )
            manager.register(spawn_verification)
            logger.info("[LoopRegistry] SpawnVerificationLoop enabled")
            return 1
        return 0
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"SpawnVerificationLoop: not available: {e}")
        failed.append("SpawnVerificationLoop")
        return 0


def _register_predictive_scaling(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    ctx: Any,
    failed: list[str],
) -> int:
    """Register PredictiveScalingLoop."""
    try:
        from scripts.p2p.loops import PredictiveScalingLoop

        predictive_scaling = PredictiveScalingLoop(
            get_role=ctx.get_role,
            get_peers=ctx.get_peers,
            get_queue_depth=ctx.get_work_queue_depth,
            get_pending_jobs_for_node=ctx.get_pending_jobs_for_node,
            spawn_preemptive_job=ctx.spawn_preemptive_job,
        )
        manager.register(predictive_scaling)
        logger.info("[LoopRegistry] PredictiveScalingLoop enabled")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"PredictiveScalingLoop: not available: {e}")
        failed.append("PredictiveScalingLoop")
        return 0


def _register_job_reassignment(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    ctx: Any,
    failed: list[str],
) -> int:
    """Register JobReassignmentLoop."""
    try:
        from scripts.p2p.loops import JobReassignmentLoop

        if ctx.job_manager is not None:
            job_reassignment = JobReassignmentLoop(
                get_role=ctx.get_role,
                check_and_reassign=ctx.job_manager.process_stale_jobs,
                get_healthy_nodes=orchestrator._get_healthy_node_ids_for_reassignment,
            )
            manager.register(job_reassignment)
            logger.info("[LoopRegistry] JobReassignmentLoop enabled")
            return 1
        return 0
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"JobReassignmentLoop: not available: {e}")
        failed.append("JobReassignmentLoop")
        return 0


def _register_auto_scaling(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register AutoScalingLoop."""
    try:
        from scripts.p2p.adapters.scale_adapters import (
            CompositeScaleAdapter,
            AutoScalingConfig,
            create_scale_adapter,
        )
        from scripts.p2p.loops import AutoScalingLoop
        from app.coordination.work_queue import get_work_queue

        scale_adapter = create_scale_adapter(
            work_queue=get_work_queue(),
            peers_getter=lambda: orchestrator.peers,
            config=AutoScalingConfig.conservative(),
        )
        orchestrator._scale_adapter = scale_adapter

        auto_scaling = AutoScalingLoop(
            get_pending_work=scale_adapter.get_pending_work,
            get_active_nodes=scale_adapter.get_active_nodes,
            get_idle_nodes=scale_adapter.get_idle_nodes,
            scale_up=scale_adapter.scale_up,
            scale_down=scale_adapter.scale_down,
        )
        manager.register(auto_scaling)
        logger.info("[LoopRegistry] AutoScalingLoop enabled")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"AutoScalingLoop: not available: {e}")
        orchestrator._scale_adapter = None
        failed.append("AutoScalingLoop")
        return 0
    except Exception as e:
        logger.warning(f"AutoScalingLoop initialization failed: {e}")
        orchestrator._scale_adapter = None
        failed.append("AutoScalingLoop")
        return 0


def _register_work_queue_maintenance(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    ctx: Any,
    failed: list[str],
) -> int:
    """Register WorkQueueMaintenanceLoop."""
    try:
        from scripts.p2p.loops import WorkQueueMaintenanceLoop

        work_queue_maint = WorkQueueMaintenanceLoop(
            is_leader=ctx.is_leader,
            get_work_queue=ctx.get_work_queue,
        )
        manager.register(work_queue_maint)
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"WorkQueueMaintenanceLoop: not available: {e}")
        failed.append("WorkQueueMaintenanceLoop")
        return 0


def _register_nat_management(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register NATManagementLoop."""
    try:
        from scripts.p2p.loops import NATManagementLoop

        nat_management = NATManagementLoop(
            detect_nat_type=orchestrator._detect_nat_type,
            probe_nat_blocked_peers=orchestrator._probe_nat_blocked_peers,
            update_relay_preferences=orchestrator._update_relay_preferences,
            validate_relay_assignments=orchestrator._validate_relay_assignments,
        )
        manager.register(nat_management)
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"NATManagementLoop: not available: {e}")
        failed.append("NATManagementLoop")
        return 0


def _register_manifest_collection(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    ctx: Any,
    failed: list[str],
) -> int:
    """Register ManifestCollectionLoop."""
    try:
        from scripts.p2p.loops import ManifestCollectionLoop

        manifest_collection = ManifestCollectionLoop(
            get_role=ctx.get_role,
            collect_cluster_manifest=orchestrator._collect_cluster_manifest,
            collect_local_manifest=orchestrator._collect_local_data_manifest,
            update_manifest=orchestrator._update_manifest_from_loop,
            update_improvement_cycle=orchestrator._update_improvement_cycle_from_loop,
            record_stats_sample=orchestrator._record_selfplay_stats_sample,
            get_alive_peers=orchestrator._get_alive_peers_for_broadcast,
            get_http_session=lambda: orchestrator.http_session,
            broadcast_enabled=True,
        )
        manager.register(manifest_collection)
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"ManifestCollectionLoop: not available: {e}")
        failed.append("ManifestCollectionLoop")
        return 0


def _register_data_management(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    ctx: Any,
    failed: list[str],
) -> int:
    """Register DataManagementLoop."""
    try:
        from scripts.p2p.loops import DataManagementLoop
        from app.utils.disk_utils import check_disk_has_capacity
        from app.db.database_repair import check_and_repair_databases

        def _check_disk_capacity_wrapper(threshold: float) -> tuple[bool, float]:
            try:
                return check_disk_has_capacity(threshold)
            except Exception as e:
                logger.debug(f"Disk capacity check error: {e}")
                return True, 0.0

        async def _check_db_integrity_wrapper(data_dir: Path) -> dict[str, int]:
            try:
                return check_and_repair_databases(
                    data_dir=data_dir,
                    auto_repair=False,
                    log_prefix="[P2P]"
                )
            except Exception as e:
                logger.debug(f"DB integrity check error: {e}")
                return {"checked": 0, "corrupted": 0, "failed": 0}

        data_management = DataManagementLoop(
            is_leader=ctx.is_leader,
            check_disk_capacity=_check_disk_capacity_wrapper,
            cleanup_disk=orchestrator._cleanup_local_disk,
            convert_jsonl_to_db=orchestrator._convert_jsonl_to_db,
            convert_jsonl_to_npz=orchestrator._convert_jsonl_to_npz_for_training,
            check_db_integrity=_check_db_integrity_wrapper,
            trigger_export=orchestrator._trigger_export_for_loop,
            start_training=orchestrator._start_auto_training,
            get_data_dir=orchestrator.get_data_directory,
            get_games_dir=lambda: orchestrator.get_data_directory() / "games",
            get_training_dir=lambda: orchestrator.get_data_directory() / "training",
            is_gpu_node=lambda: orchestrator.self_info.is_gpu_node() if orchestrator.self_info else False,
            has_training_jobs=lambda: (orchestrator.self_info.training_jobs > 0) if orchestrator.self_info else False,
        )
        manager.register(data_management)
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"DataManagementLoop: not available: {e}")
        failed.append("DataManagementLoop")
        return 0


def _register_model_sync(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register ModelSyncLoop."""
    try:
        from scripts.p2p.loops import ModelSyncLoop

        # Check if model sync is available
        HAS_MODEL_SYNC = hasattr(orchestrator, '_model_versions')
        HAS_HOSTS_FOR_SYNC = hasattr(orchestrator, 'cluster_config')

        if not (HAS_MODEL_SYNC and HAS_HOSTS_FOR_SYNC):
            logger.debug("ModelSyncLoop: skipped (model sync not available)")
            return 0

        async def _get_node_models_for_sync(node_id: str) -> dict[str, str]:
            if hasattr(orchestrator, 'cluster_data_manifest') and orchestrator.cluster_data_manifest:
                node_data = orchestrator.cluster_data_manifest.get(node_id, {})
                models = node_data.get("models", {})
                return {k: v.get("version", "") for k, v in models.items()} if isinstance(models, dict) else {}
            return {}

        async def _sync_model_to_node(node_id: str, config_key: str, source_path: str) -> bool:
            try:
                peer = orchestrator._peer_snapshot.get_snapshot().get(node_id)
                if not peer:
                    return False
                host = getattr(peer, "ip", None) or getattr(peer, "host", "")
                if not host:
                    return False
                cmd = [
                    "rsync", "-az", "--timeout=60",
                    source_path,
                    f"{host}:{source_path}",
                ]
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
                return proc.returncode == 0
            except (asyncio.TimeoutError, OSError, subprocess.SubprocessError, ValueError):
                return False

        model_sync = ModelSyncLoop(
            get_model_versions=lambda: getattr(orchestrator, '_model_versions', {}),
            get_node_models=_get_node_models_for_sync,
            sync_model=_sync_model_to_node,
            get_active_nodes=lambda: list(orchestrator._peer_snapshot.get_snapshot().keys()),
        )
        manager.register(model_sync)
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"ModelSyncLoop: not available: {e}")
        failed.append("ModelSyncLoop")
        return 0


def _register_model_fetch(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    ctx: Any,
    failed: list[str],
) -> int:
    """Register ModelFetchLoop."""
    try:
        from scripts.p2p.loops.data_loops import ModelFetchLoop

        _fetched_model_jobs: set[str] = set()

        def _get_completed_training_jobs() -> list:
            with orchestrator.training_jobs_lock:
                return [
                    job for job in orchestrator.training_jobs.values()
                    if getattr(job, "status", "") == "completed"
                ]

        async def _fetch_model_for_job(job) -> bool:
            try:
                return await orchestrator.training_coordinator._fetch_model_from_training_node(job)
            except (AttributeError, TypeError) as e:
                logger.debug(f"ModelFetchLoop: fetch error: {e}")
                return False

        def _mark_model_fetched(job_id: str) -> None:
            _fetched_model_jobs.add(job_id)

        def _is_model_fetched(job_id: str) -> bool:
            return job_id in _fetched_model_jobs

        model_fetch = ModelFetchLoop(
            is_leader=ctx.is_leader,
            get_completed_training_jobs=_get_completed_training_jobs,
            fetch_model=_fetch_model_for_job,
            mark_model_fetched=_mark_model_fetched,
            is_model_fetched=_is_model_fetched,
        )
        manager.register(model_fetch)
        logger.info("[LoopRegistry] ModelFetchLoop registered")
        return 1
    except (ImportError, TypeError, ValueError, AttributeError) as e:
        logger.debug(f"ModelFetchLoop: not available: {e}")
        failed.append("ModelFetchLoop")
        return 0


def _register_validation(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    ctx: Any,
    failed: list[str],
) -> int:
    """Register ValidationLoop."""
    try:
        from scripts.p2p.loops import ValidationLoop
        from app.coordination.work_queue import get_work_queue

        def _get_model_registry():
            try:
                from app.training.model_registry import ModelRegistry
                return ModelRegistry()
            except (ImportError, TypeError, ValueError) as e:
                logger.debug(f"ValidationLoop: registry not available: {e}")
                return None

        async def _send_validation_notification(msg: str, severity: str, context: dict) -> None:
            await orchestrator.notifier.send(msg, severity=severity, context=context)

        validation = ValidationLoop(
            is_leader=ctx.is_leader,
            get_model_registry=_get_model_registry,
            get_work_queue=get_work_queue,
            send_notification=_send_validation_notification,
        )
        manager.register(validation)
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"ValidationLoop: not available: {e}")
        failed.append("ValidationLoop")
        return 0


def _register_tailscale_peer_discovery(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    ctx: Any,
    failed: list[str],
) -> int:
    """Register TailscalePeerDiscoveryLoop."""
    try:
        from scripts.p2p.loops import TailscalePeerDiscoveryLoop

        def _get_current_peer_ids() -> set[str]:
            return {p.node_id for p in orchestrator._peer_snapshot.get_snapshot().values()}

        def _get_alive_peer_count() -> int:
            return sum(1 for p in orchestrator._peer_snapshot.get_snapshot().values() if p.is_alive())

        async def _probe_and_connect_peer(ip: str, hostname: str) -> bool:
            try:
                url = f"http://{ip}:{DEFAULT_PORT}/health"
                timeout = ClientTimeout(total=10)
                async with _get_client_session(timeout) as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            node_id = data.get("node_id", hostname)
                            logger.debug(f"TailscalePeerDiscovery: connected to {node_id}")
                            await orchestrator._send_heartbeat_to_peer(ip, DEFAULT_PORT)
                            return True
                return False
            except Exception as e:
                logger.debug(f"TailscalePeerDiscovery: failed to connect to {hostname}: {e}")
                return False

        ts_peer_discovery = TailscalePeerDiscoveryLoop(
            is_leader=ctx.is_leader,
            get_current_peers=_get_current_peer_ids,
            get_alive_peer_count=_get_alive_peer_count,
            probe_and_connect=_probe_and_connect_peer,
        )
        manager.register(ts_peer_discovery)
        logger.info("[LoopRegistry] TailscalePeerDiscoveryLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.warning(f"TailscalePeerDiscoveryLoop: not available: {e}")
        failed.append("TailscalePeerDiscoveryLoop")
        return 0


def _register_peer_cleanup(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register PeerCleanupLoop."""
    try:
        from scripts.p2p.loops import PeerCleanupLoop, PeerCleanupConfig

        async def _purge_single_peer_async(node_id: str) -> bool:
            with orchestrator.peers_lock:
                if node_id in orchestrator.peers:
                    del orchestrator.peers[node_id]
                    logger.debug(f"[PeerCleanup] Removed peer: {node_id}")
                    orchestrator._sync_peer_snapshot()
                    return True
            return False

        peer_cleanup = PeerCleanupLoop(
            get_all_peers=lambda: dict(orchestrator.peers),
            purge_peer=_purge_single_peer_async,
            config=PeerCleanupConfig(
                cleanup_interval_seconds=float(
                    os.environ.get("RINGRIFT_PEER_CLEANUP_INTERVAL", "300")
                ),
                enabled=os.environ.get(
                    "RINGRIFT_PEER_CLEANUP_ENABLED", "1"
                ).lower() in ("1", "true", "yes", "on"),
            ),
        )
        manager.register(peer_cleanup)
        logger.info("[LoopRegistry] PeerCleanupLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.warning(f"PeerCleanupLoop: not available: {e}")
        failed.append("PeerCleanupLoop")
        return 0


def _register_gossip_state_cleanup(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register GossipStateCleanupLoop."""
    try:
        from scripts.p2p.loops import GossipStateCleanupLoop, GossipStateCleanupConfig

        gossip_cleanup = GossipStateCleanupLoop(
            get_orchestrator=lambda: orchestrator,
            emit_event=orchestrator._safe_emit_p2p_event,
            config=GossipStateCleanupConfig(),
        )
        manager.register(gossip_cleanup)
        logger.info("[LoopRegistry] GossipStateCleanupLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.warning(f"GossipStateCleanupLoop: not available: {e}")
        failed.append("GossipStateCleanupLoop")
        return 0


def _register_gossip_peer_promotion(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register GossipPeerPromotionLoop."""
    try:
        from scripts.p2p.loops.gossip_peer_promotion_loop import (
            GossipPeerPromotionLoop,
            GossipPeerPromotionConfig,
        )

        gossip_promotion = GossipPeerPromotionLoop(
            orchestrator=orchestrator,
            config=GossipPeerPromotionConfig(
                interval_seconds=30.0,
                max_peers_per_cycle=5,
                min_retry_interval_seconds=60.0,
            ),
        )
        manager.register(gossip_promotion)
        logger.info("[LoopRegistry] GossipPeerPromotionLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.warning(f"GossipPeerPromotionLoop: not available: {e}")
        failed.append("GossipPeerPromotionLoop")
        return 0


def _register_quorum_crisis_discovery(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register QuorumCrisisDiscoveryLoop."""
    try:
        from scripts.p2p.loops import QuorumCrisisDiscoveryLoop, QuorumCrisisConfig

        def _get_bootstrap_seeds_for_crisis() -> list[str]:
            seeds = getattr(orchestrator, "bootstrap_seeds", [])
            return list(seeds) if seeds else []

        def _get_voter_endpoints_for_crisis() -> list[tuple[str, int]]:
            voters = []
            for peer_id, peer_info in orchestrator.state_manager.get_peers().items():
                if hasattr(peer_info, "is_voter") and peer_info.is_voter:
                    ip = getattr(peer_info, "tailscale_ip", None) or getattr(peer_info, "ip", None)
                    port = getattr(peer_info, "port", 8770)
                    if ip:
                        voters.append((ip, port))
            return voters

        async def _probe_endpoint_for_crisis(addr: str) -> bool:
            try:
                return await orchestrator._probe_peer_health(addr)
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                return False

        async def _on_peer_discovered_during_crisis(peer_id: str, addr: str) -> None:
            logger.info(f"[QuorumCrisis] Peer discovered: {peer_id} @ {addr}")
            if peer_id not in orchestrator.state_manager.get_peers():
                await orchestrator._bootstrap_from_peer(addr)

        quorum_crisis = QuorumCrisisDiscoveryLoop(
            get_bootstrap_seeds=_get_bootstrap_seeds_for_crisis,
            get_voter_endpoints=_get_voter_endpoints_for_crisis,
            probe_endpoint=_probe_endpoint_for_crisis,
            on_peer_discovered=_on_peer_discovered_during_crisis,
            emit_event=orchestrator._safe_emit_p2p_event,
            config=QuorumCrisisConfig(),
        )
        manager.register(quorum_crisis)
        orchestrator._quorum_crisis_loop = quorum_crisis
        logger.info("[LoopRegistry] QuorumCrisisDiscoveryLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.warning(f"QuorumCrisisDiscoveryLoop: not available: {e}")
        failed.append("QuorumCrisisDiscoveryLoop")
        return 0


def _register_worker_pull(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    ctx: Any,
    failed: list[str],
) -> int:
    """Register WorkerPullLoop."""
    try:
        from scripts.p2p.loops import WorkerPullLoop
        from app.coordination.work_queue import get_work_queue

        def _get_self_metrics_for_pull() -> dict[str, Any]:
            return {
                "gpu_percent": getattr(orchestrator.self_info, "gpu_percent", 0),
                "cpu_percent": getattr(orchestrator.self_info, "cpu_percent", 0),
                "training_jobs": getattr(orchestrator.self_info, "training_jobs", 0),
                "has_gpu": getattr(orchestrator.self_info, "has_gpu", False),
                "selfplay_jobs": getattr(orchestrator.self_info, "selfplay_jobs", 0),
                "max_selfplay_slots": getattr(orchestrator.self_info, "max_selfplay_slots", 8),
            }

        async def _pop_autonomous_work_for_pull() -> dict[str, Any] | None:
            loop = getattr(orchestrator, "_autonomous_queue_loop", None)
            if loop and hasattr(loop, "pop_local_work"):
                try:
                    return await loop.pop_local_work()
                except Exception as e:
                    logger.debug(f"[WorkerPull] Autonomous queue pop failed: {e}")
            return None

        def _get_work_discovery_manager_for_pull():
            try:
                from scripts.p2p.managers.work_discovery_manager import get_work_discovery_manager
                return get_work_discovery_manager()
            except ImportError:
                return None

        def _set_work_queue_partition_state(is_partitioned: bool) -> None:
            wq = get_work_queue()
            if wq and hasattr(wq, "set_partition_state"):
                try:
                    wq.set_partition_state(is_partitioned)
                except Exception as e:
                    logger.debug(f"[WorkerPull] Failed to set partition state: {e}")

        async def _probe_leader_health_for_pull(leader_id: str) -> dict[str, Any] | None:
            with orchestrator.peers_lock:
                peer = orchestrator.peers.get(leader_id)
                if not peer:
                    return None
                host = peer.host
                port = peer.port

            if not host or not port:
                return None

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{host}:{port}/status",
                        timeout=aiohttp.ClientTimeout(total=5.0),
                    ) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        return None
            except Exception as e:
                logger.debug(f"[WorkerPull] Leader {leader_id} probe failed: {e}")
                return None

        def _get_allowed_work_types_for_pull() -> list[str]:
            """Return work types this node is allowed to claim.

            Feb 2026: Prevents coordinator from claiming selfplay/training work
            that it will then skip, wasting queue slots for GPU nodes.
            """
            from scripts.p2p.managers.work_discovery_manager import (
                _is_selfplay_enabled_for_node,
                _is_training_enabled_for_node,
            )

            all_types = ["selfplay", "training", "gpu_cmaes", "tournament", "gauntlet"]
            allowed = []
            for work_type in all_types:
                if work_type == "selfplay" and not _is_selfplay_enabled_for_node():
                    continue
                if work_type in ("training", "gpu_cmaes") and not _is_training_enabled_for_node():
                    continue
                allowed.append(work_type)
            return allowed

        worker_pull = WorkerPullLoop(
            is_leader=ctx.is_leader,
            get_leader_id=ctx.get_leader_id,
            get_self_metrics=_get_self_metrics_for_pull,
            claim_work_from_leader=orchestrator._claim_work_from_leader,
            execute_work=orchestrator._execute_claimed_work,
            report_work_result=orchestrator._report_work_result,
            get_allowed_work_types=_get_allowed_work_types_for_pull,
            pop_autonomous_work=_pop_autonomous_work_for_pull,
            get_work_discovery_manager=_get_work_discovery_manager_for_pull,
            claim_work_batch_from_leader=orchestrator._claim_work_batch_from_leader,
            probe_leader_health=_probe_leader_health_for_pull,
            set_partition_state=_set_work_queue_partition_state,
        )
        manager.register(worker_pull)
        logger.info("[LoopRegistry] WorkerPullLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.error(f"[LoopRegistry] WorkerPullLoop registration FAILED: {e}")
        failed.append("WorkerPullLoop")
        return 0


def _register_follower_discovery(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    ctx: Any,
    failed: list[str],
) -> int:
    """Register FollowerDiscoveryLoop."""
    try:
        from scripts.p2p.loops import FollowerDiscoveryLoop

        def _get_known_peer_addresses() -> list[str]:
            with orchestrator.peers_lock:
                peers_snapshot = list(orchestrator.peers.values())
            addresses = [
                f"{p.host}:{p.port}"
                for p in peers_snapshot
                if p.is_alive() and p.host and p.port
            ]
            if not addresses and orchestrator.known_peers:
                addresses = list(orchestrator.known_peers)
            return addresses

        async def _query_peer_list(peer_addr: str) -> list[str] | None:
            try:
                host, port_str = peer_addr.rsplit(":", 1)
                port = int(port_str)
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{host}:{port}/peers",
                        timeout=aiohttp.ClientTimeout(total=5.0),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data.get("peers", [])
            except (aiohttp.ClientError, ValueError, asyncio.TimeoutError):
                pass
            return None

        def _add_discovered_peer(peer_addr: str) -> None:
            try:
                host, port_str = peer_addr.rsplit(":", 1)
                port = int(port_str)
                asyncio.create_task(
                    orchestrator._send_heartbeat_to_peer(host, port),
                    name=f"discover_peer_{peer_addr}",
                )
            except ValueError:
                logger.debug(f"Invalid peer address format: {peer_addr}")

        follower_discovery = FollowerDiscoveryLoop(
            get_known_peers=_get_known_peer_addresses,
            query_peer_list=_query_peer_list,
            add_peer=_add_discovered_peer,
            is_leader=ctx.is_leader,
        )
        manager.register(follower_discovery)
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.warning(f"FollowerDiscoveryLoop: not available: {e}")
        failed.append("FollowerDiscoveryLoop")
        return 0


def _register_self_healing(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    ctx: Any,
    failed: list[str],
) -> int:
    """Register SelfHealingLoop."""
    try:
        from scripts.p2p.loops import SelfHealingLoop
        from app.coordination.health_manager import get_health_manager
        from app.coordination.work_queue import get_work_queue

        async def _restart_stopped_loops_callback() -> dict[str, bool]:
            lm = orchestrator._get_loop_manager()
            if lm is not None:
                return await lm.restart_stopped_loops()
            return {}

        self_healing = SelfHealingLoop(
            is_leader=ctx.is_leader,
            get_health_manager=get_health_manager,
            get_work_queue=get_work_queue,
            cleanup_stale_processes=orchestrator._cleanup_stale_processes,
            restart_stopped_loops=_restart_stopped_loops_callback,
        )
        manager.register(self_healing)
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"SelfHealingLoop: not available: {e}")
        failed.append("SelfHealingLoop")
        return 0


def _register_remote_p2p_recovery(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    ctx: Any,
    failed: list[str],
) -> int:
    """Register RemoteP2PRecoveryLoop."""
    try:
        from scripts.p2p.loops import RemoteP2PRecoveryLoop

        def _get_alive_peer_ids_for_recovery() -> list[str]:
            with orchestrator.peers_lock:
                peers_snapshot = list(orchestrator.peers.values())
            return [p.node_id for p in peers_snapshot if p.is_alive()]

        remote_recovery = RemoteP2PRecoveryLoop(
            get_alive_peer_ids=_get_alive_peer_ids_for_recovery,
            is_leader=ctx.is_leader,
        )
        manager.register(remote_recovery)
        logger.info(f"[LoopRegistry] RemoteP2PRecoveryLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"RemoteP2PRecoveryLoop: not available: {e}")
        failed.append("RemoteP2PRecoveryLoop")
        return 0


def _register_leader_probe(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register LeaderProbeLoop."""
    try:
        from scripts.p2p.loops import LeaderProbeLoop
        leader_probe = LeaderProbeLoop(orchestrator=orchestrator)
        manager.register(leader_probe)
        logger.info("[LoopRegistry] LeaderProbeLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"LeaderProbeLoop: not available: {e}")
        failed.append("LeaderProbeLoop")
        return 0


def _register_leader_maintenance(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register LeaderMaintenanceLoop."""
    try:
        from scripts.p2p.loops import LeaderMaintenanceLoop
        leader_maintenance = LeaderMaintenanceLoop(orchestrator=orchestrator)
        manager.register(leader_maintenance)
        logger.info("[LoopRegistry] LeaderMaintenanceLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"LeaderMaintenanceLoop: not available: {e}")
        failed.append("LeaderMaintenanceLoop")
        return 0


def _register_relay_health(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register RelayHealthLoop."""
    try:
        from scripts.p2p.loops import RelayHealthLoop

        def _get_relay_nodes_for_loop() -> list[str]:
            relay_nodes = []
            # Use getattr with None fallback to safely access cached cluster config
            cluster_config = getattr(orchestrator, "_cluster_config", None)
            if cluster_config:
                hosts = cluster_config.get("hosts", {})
                for node_id, config in hosts.items():
                    if config.get("relay_capable", False) and config.get("status") == "active":
                        relay_nodes.append(node_id)
            return relay_nodes or ["hetzner-cpu1", "hetzner-cpu2", "hetzner-cpu3"]

        def _get_node_info_for_relay(node_id: str) -> dict[str, Any] | None:
            cluster_config = getattr(orchestrator, "_cluster_config", None)
            if cluster_config:
                hosts = cluster_config.get("hosts", {})
                return hosts.get(node_id)
            return None

        def _get_nat_blocked_peers_for_relay() -> dict[str, str]:
            result: dict[str, str] = {}
            cluster_config = getattr(orchestrator, "_cluster_config", None)
            if cluster_config:
                hosts = cluster_config.get("hosts", {})
                for node_id, config in hosts.items():
                    relay_primary = config.get("relay_primary")
                    if relay_primary and config.get("status") == "active":
                        result[node_id] = relay_primary
            return result

        async def _trigger_relay_failover_for_loop(
            peer_id: str, old_relay: str, new_relay: str
        ) -> bool:
            logger.info(
                f"[RelayHealth] Failover recommended: {peer_id} "
                f"should switch from {old_relay} to {new_relay}"
            )
            orchestrator._safe_emit_event("RELAY_FAILOVER_RECOMMENDED", {
                "peer_id": peer_id,
                "old_relay": old_relay,
                "new_relay": new_relay,
            })
            return True

        relay_health = RelayHealthLoop(
            get_relay_nodes=_get_relay_nodes_for_loop,
            get_node_info=_get_node_info_for_relay,
            get_nat_blocked_peers=_get_nat_blocked_peers_for_relay,
            trigger_relay_failover=_trigger_relay_failover_for_loop,
            emit_event=orchestrator._safe_emit_event,
        )
        manager.register(relay_health)
        logger.info("[LoopRegistry] RelayHealthLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"RelayHealthLoop: not available: {e}")
        failed.append("RelayHealthLoop")
        return 0


def _register_predictive_monitoring(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register PredictiveMonitoringLoop."""
    try:
        from scripts.p2p.loops import PredictiveMonitoringLoop
        from scripts.p2p_orchestrator import NodeRole
        from app.coordination.predictive_alerts import get_predictive_alerts
        from app.coordination.work_queue import get_work_queue

        def _get_peers_for_monitoring() -> list[Any]:
            with orchestrator.peers_lock:
                peers_snapshot = list(orchestrator.peers.values())
            return [p for p in peers_snapshot if p.is_alive()]

        def _get_production_models() -> tuple[list[str], float]:
            model_ids = []
            last_training = time.time() - 3600
            try:
                from app.training.model_registry import ModelRegistry, ModelStage
                registry = ModelRegistry()
                production_models = registry.get_versions_by_stage(ModelStage.PRODUCTION)
                model_ids = [f"{m['model_id']}_v{m['version']}" for m in production_models]
                if production_models:
                    from datetime import datetime
                    latest_update = max(
                        datetime.fromisoformat(m['updated_at'].replace('Z', '+00:00'))
                        for m in production_models
                        if m.get('updated_at')
                    )
                    last_training = latest_update.timestamp()
            except Exception as e:
                logger.debug(f"Model registry lookup failed: {e}")
            return model_ids, last_training

        predictive_monitoring = PredictiveMonitoringLoop(
            is_leader=lambda: orchestrator.role == NodeRole.LEADER,
            get_alert_manager=get_predictive_alerts,
            get_work_queue=get_work_queue,
            get_peers=_get_peers_for_monitoring,
            get_notifier=lambda: orchestrator.notifier,
            get_production_models=_get_production_models,
        )
        manager.register(predictive_monitoring)
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"PredictiveMonitoringLoop: not available: {e}")
        failed.append("PredictiveMonitoringLoop")
        return 0


def _register_training_sync(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    ctx: Any,
    failed: list[str],
) -> int:
    """Register TrainingSyncLoop."""
    try:
        from scripts.p2p.loops import TrainingSyncLoop
        from app.utils.disk_utils import check_disk_has_capacity

        training_sync = TrainingSyncLoop(
            is_leader=ctx.is_leader,
            sync_to_training_nodes=orchestrator._sync_selfplay_to_training_nodes,
            get_last_sync_time=lambda: getattr(orchestrator, 'last_training_sync_time', 0.0),
            check_disk_capacity=lambda: check_disk_has_capacity(70.0),
        )
        manager.register(training_sync)
        logger.info("[LoopRegistry] TrainingSyncLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.warning(f"TrainingSyncLoop: not available: {e}")
        failed.append("TrainingSyncLoop")
        return 0


def _register_data_aggregation(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register DataAggregationLoop."""
    try:
        from scripts.p2p.loops import DataAggregationLoop

        def _get_node_game_counts_for_aggregation() -> dict[str, int]:
            result: dict[str, int] = {}
            if hasattr(orchestrator, 'cluster_data_manifest') and orchestrator.cluster_data_manifest:
                for node_id, node_data in orchestrator.cluster_data_manifest.node_manifests.items():
                    if hasattr(node_data, 'selfplay_games'):
                        result[node_id] = node_data.selfplay_games
                    elif isinstance(node_data, dict):
                        result[node_id] = node_data.get('selfplay_games', 0)
                    else:
                        result[node_id] = 0
            return result

        async def _aggregate_from_node_for_loop(node_id: str) -> dict[str, Any]:
            try:
                peer = orchestrator.peers.get(node_id)
                if not peer:
                    return {"success": False, "error": "peer_not_found"}
                if hasattr(orchestrator, 'sync_planner') and orchestrator.sync_planner:
                    result = await orchestrator.sync_planner.sync_from_node(node_id)
                    return {"success": result, "node_id": node_id}
                return {"success": False, "error": "sync_planner_unavailable"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        data_aggregation = DataAggregationLoop(
            get_node_game_counts=_get_node_game_counts_for_aggregation,
            aggregate_from_node=_aggregate_from_node_for_loop,
        )
        manager.register(data_aggregation)
        logger.info("[LoopRegistry] DataAggregationLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.warning(f"DataAggregationLoop: not available: {e}")
        failed.append("DataAggregationLoop")
        return 0


def _register_health_aggregation(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register HealthAggregationLoop."""
    try:
        from scripts.p2p.loops import HealthAggregationLoop

        def _get_node_ids_for_health() -> list[str]:
            with orchestrator.peers_lock:
                peers_snapshot = list(orchestrator.peers.values())
            return [p.node_id for p in peers_snapshot if p.is_alive()]

        async def _fetch_node_health_for_loop(node_id: str) -> dict[str, Any]:
            try:
                peer = orchestrator._peer_snapshot.get_snapshot().get(node_id)
                if not peer:
                    return {"healthy": False, "error": "peer_not_found"}
                host = peer.host or peer.ip
                port = peer.port or DEFAULT_PORT
                url = f"http://{host}:{port}/health"
                timeout = ClientTimeout(total=10)
                async with _get_client_session(timeout) as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return {
                                "healthy": True,
                                "cpu_percent": data.get("cpu_percent", 0),
                                "memory_percent": data.get("memory_percent", 0),
                                "gpu_percent": data.get("gpu_percent", 0),
                                "disk_percent": data.get("disk_percent", 0),
                            }
                        return {"healthy": False, "error": f"status_{resp.status}"}
            except Exception as e:
                return {"healthy": False, "error": str(e)}

        def _on_health_updated_callback(health_data: dict[str, dict[str, Any]]) -> None:
            orchestrator._aggregated_node_health = health_data

        def _recover_unhealthy_nodes_callback() -> list[str]:
            if hasattr(orchestrator, "node_selector") and orchestrator.node_selector:
                return orchestrator.node_selector.recover_unhealthy_nodes()
            return []

        health_aggregation = HealthAggregationLoop(
            get_node_ids=_get_node_ids_for_health,
            fetch_node_health=_fetch_node_health_for_loop,
            on_health_updated=_on_health_updated_callback,
            on_recover_unhealthy=_recover_unhealthy_nodes_callback,
        )
        manager.register(health_aggregation)
        logger.info("[LoopRegistry] HealthAggregationLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.warning(f"HealthAggregationLoop: not available: {e}")
        failed.append("HealthAggregationLoop")
        return 0


def _register_ip_discovery(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register IpDiscoveryLoop."""
    try:
        from scripts.p2p.loops import IpDiscoveryLoop

        def _get_nodes_for_ip_discovery() -> dict[str, dict[str, Any]]:
            with orchestrator.peers_lock:
                return {
                    p.node_id: {
                        "ip": p.host or p.ip,
                        "tailscale_ip": getattr(p, "tailscale_ip", None),
                        "public_ip": getattr(p, "public_ip", None),
                        "hostname": getattr(p, "hostname", None),
                    }
                    for p in orchestrator.peers.values()
                }

        async def _update_node_ip_for_discovery(node_id: str, new_ip: str) -> None:
            with orchestrator.peers_lock:
                if node_id in orchestrator.peers:
                    peer = orchestrator.peers[node_id]
                    peer.host = new_ip
                    peer.ip = new_ip
                    logger.info(f"[IpDiscovery] Updated {node_id} IP to {new_ip}")

        ip_discovery = IpDiscoveryLoop(
            get_nodes=_get_nodes_for_ip_discovery,
            update_node_ip=_update_node_ip_for_discovery,
        )
        manager.register(ip_discovery)
        logger.info("[LoopRegistry] IpDiscoveryLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"IpDiscoveryLoop: not available: {e}")
        failed.append("IpDiscoveryLoop")
        return 0


def _register_tailscale_recovery(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register TailscaleRecoveryLoop."""
    try:
        from scripts.p2p.loops import TailscaleRecoveryLoop

        def _get_tailscale_status_for_recovery() -> dict[str, Any]:
            result = {}
            with orchestrator.peers_lock:
                for p in orchestrator.peers.values():
                    result[p.node_id] = {
                        "tailscale_state": getattr(p, "tailscale_state", "unknown"),
                        "tailscale_online": getattr(p, "tailscale_online", True),
                    }
            return result

        async def _run_ssh_for_tailscale_recovery(node_id: str, cmd: str) -> Any:
            try:
                peer = orchestrator.peers.get(node_id)
                if not peer:
                    return type("Result", (), {"returncode": 1, "stdout": "", "stderr": "peer not found"})()
                host = peer.host or peer.ip
                proc = await asyncio.create_subprocess_exec(
                    "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
                    f"root@{host}", cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                return type("Result", (), {"returncode": proc.returncode, "stdout": stdout.decode(), "stderr": stderr.decode()})()
            except (asyncio.TimeoutError, OSError) as e:
                return type("Result", (), {"returncode": 1, "stdout": "", "stderr": str(e)})()

        tailscale_recovery = TailscaleRecoveryLoop(
            get_tailscale_status=_get_tailscale_status_for_recovery,
            run_ssh_command=_run_ssh_for_tailscale_recovery,
        )
        manager.register(tailscale_recovery)
        logger.info("[LoopRegistry] TailscaleRecoveryLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"TailscaleRecoveryLoop: not available: {e}")
        failed.append("TailscaleRecoveryLoop")
        return 0


def _register_tailscale_keepalive(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register TailscaleKeepaliveLoop."""
    try:
        from scripts.p2p.loops import TailscaleKeepaliveLoop, TailscaleKeepaliveConfig

        def _get_peer_tailscale_ips_for_keepalive() -> dict[str, str]:
            result = {}
            with orchestrator.peers_lock:
                for p in orchestrator.peers.values():
                    ts_ip = getattr(p, "tailscale_ip", None)
                    if ts_ip and ts_ip != "0.0.0.0":
                        result[p.node_id] = ts_ip
            return result

        def _is_userspace_mode() -> bool:
            try:
                return not os.path.exists("/dev/net/tun")
            except OSError:
                return True

        async def _on_connection_quality_change(node_id: str, ip: str, is_direct: bool) -> None:
            conn_type = "direct" if is_direct else "DERP relay"
            logger.info(f"[TailscaleKeepalive] {node_id} ({ip}): now using {conn_type}")

        keepalive_config = TailscaleKeepaliveConfig(
            interval_seconds=60.0,
            userspace_interval_seconds=30.0,
            derp_recovery_interval_seconds=180.0,
        )

        tailscale_keepalive = TailscaleKeepaliveLoop(
            get_peer_tailscale_ips=_get_peer_tailscale_ips_for_keepalive,
            is_userspace_mode=_is_userspace_mode,
            on_connection_quality_change=_on_connection_quality_change,
            config=keepalive_config,
        )
        manager.register(tailscale_keepalive)
        is_userspace = _is_userspace_mode()
        logger.info(
            f"[LoopRegistry] TailscaleKeepaliveLoop registered "
            f"(userspace_mode={is_userspace})"
        )
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"TailscaleKeepaliveLoop: not available: {e}")
        failed.append("TailscaleKeepaliveLoop")
        return 0


def _register_cluster_healing(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register ClusterHealingLoop."""
    try:
        from scripts.p2p.loops import ClusterHealingLoop, ClusterHealingConfig

        def _get_current_peer_ids() -> set[str]:
            with orchestrator.peers_lock:
                return set(orchestrator.peers.keys())

        def _get_alive_peer_addresses() -> list[str]:
            with orchestrator.peers_lock:
                peers_snapshot = list(orchestrator.peers.values())
            return [
                f"http://{p.tailscale_ip or p.host or p.ip}:{p.port or DEFAULT_PORT}"
                for p in peers_snapshot
                if p.is_alive() and (p.tailscale_ip or p.host or p.ip)
            ]

        def _on_node_joined(node_id: str) -> None:
            logger.info(f"[ClusterHealing] Node {node_id} joined the cluster")

        healing_config = ClusterHealingConfig(
            check_interval_seconds=300.0,
            max_heal_per_cycle=5,
            ssh_timeout_seconds=30.0,
            p2p_startup_wait_seconds=15.0,
        )

        cluster_healing = ClusterHealingLoop(
            get_current_peers=_get_current_peer_ids,
            get_alive_peer_addresses=_get_alive_peer_addresses,
            emit_event=orchestrator._safe_emit_event if hasattr(orchestrator, "_safe_emit_event") else None,
            on_node_joined=_on_node_joined,
            config=healing_config,
        )
        manager.register(cluster_healing)
        logger.info("[LoopRegistry] ClusterHealingLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"ClusterHealingLoop: not available: {e}")
        failed.append("ClusterHealingLoop")
        return 0


def _register_udp_discovery(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register UdpDiscoveryLoop."""
    try:
        from scripts.p2p.loops import UdpDiscoveryLoop

        def _get_known_peer_addrs_for_udp() -> list[str]:
            with orchestrator.peers_lock:
                return [
                    f"{p.host or p.ip}:{p.port or DEFAULT_PORT}"
                    for p in orchestrator.peers.values()
                    if p.host or p.ip
                ]

        def _add_peer_from_udp_discovery(peer_addr: str) -> None:
            try:
                host, port_str = peer_addr.rsplit(":", 1)
                port = int(port_str)
                asyncio.create_task(
                    orchestrator._send_heartbeat_to_peer(host, port),
                    name=f"udp_discover_{peer_addr}",
                )
            except ValueError:
                logger.debug(f"[UdpDiscovery] Invalid peer address: {peer_addr}")

        udp_discovery = UdpDiscoveryLoop(
            get_node_id=lambda: orchestrator.node_id,
            get_host=lambda: orchestrator.host or "0.0.0.0",
            get_port=lambda: orchestrator.port,
            get_known_peers=_get_known_peer_addrs_for_udp,
            add_peer=_add_peer_from_udp_discovery,
        )
        manager.register(udp_discovery)
        logger.info("[LoopRegistry] UdpDiscoveryLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"UdpDiscoveryLoop: not available: {e}")
        failed.append("UdpDiscoveryLoop")
        return 0


def _register_split_brain_detection(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register SplitBrainDetectionLoop."""
    try:
        from scripts.p2p.loops import SplitBrainDetectionLoop

        def _get_peer_endpoint_for_split_brain(peer_id: str) -> str | None:
            peer = orchestrator._peer_snapshot.get_snapshot().get(peer_id)
            if not peer:
                return None
            host = peer.host or peer.ip
            port = peer.port or DEFAULT_PORT
            return f"http://{host}:{port}"

        async def _on_split_brain_detected_callback(leaders: list[str], epoch: int) -> None:
            logger.critical(
                f"[SplitBrain] DETECTED: {len(leaders)} leaders in cluster: {leaders} "
                f"(epoch={epoch})"
            )
            await orchestrator._emit_split_brain_detected(
                detected_leaders=leaders,
                resolution_action="election",
            )

        split_brain_detection = SplitBrainDetectionLoop(
            get_peers=lambda: dict(orchestrator._peer_snapshot.get_snapshot()),
            get_peer_endpoint=_get_peer_endpoint_for_split_brain,
            get_own_leader_id=lambda: orchestrator.leader_id,
            get_cluster_epoch=lambda: getattr(orchestrator, "cluster_epoch", 0),
            on_split_brain_detected=_on_split_brain_detected_callback,
        )
        manager.register(split_brain_detection)
        logger.info("[LoopRegistry] SplitBrainDetectionLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"SplitBrainDetectionLoop: not available: {e}")
        failed.append("SplitBrainDetectionLoop")
        return 0


def _register_autonomous_queue_population(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register AutonomousQueuePopulationLoop."""
    try:
        from scripts.p2p.loops import AutonomousQueuePopulationLoop

        autonomous_queue = AutonomousQueuePopulationLoop(
            orchestrator=orchestrator,
            config=None,
        )
        manager.register(autonomous_queue)
        orchestrator._autonomous_queue_loop = autonomous_queue
        logger.info("[LoopRegistry] AutonomousQueuePopulationLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"AutonomousQueuePopulationLoop: not available: {e}")
        orchestrator._autonomous_queue_loop = None
        failed.append("AutonomousQueuePopulationLoop")
        return 0


def _register_http_server_health(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register HttpServerHealthLoop."""
    try:
        from scripts.p2p.loops import HttpServerHealthLoop

        http_health = HttpServerHealthLoop(
            port=orchestrator.port,
            restart_callback=orchestrator.restart_http_server,
        )
        manager.register(http_health)
        logger.info("[LoopRegistry] HttpServerHealthLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"HttpServerHealthLoop: not available: {e}")
        failed.append("HttpServerHealthLoop")
        return 0


def _register_comprehensive_evaluation(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register ComprehensiveEvaluationLoop."""
    try:
        from scripts.p2p.loops import (
            ComprehensiveEvaluationLoop,
            ComprehensiveEvaluationConfig,
        )

        comp_eval_loop = ComprehensiveEvaluationLoop(
            get_role=lambda: orchestrator.role,
            get_orchestrator=lambda: orchestrator,
            config=ComprehensiveEvaluationConfig(
                interval=6 * 3600,
                max_evaluations_per_cycle=50,
                stale_threshold_days=7,
                games_per_harness=50,
                save_games=True,
                register_with_elo=True,
            ),
        )
        manager.register(comp_eval_loop)
        logger.info("[LoopRegistry] ComprehensiveEvaluationLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"ComprehensiveEvaluationLoop: not available: {e}")
        failed.append("ComprehensiveEvaluationLoop")
        return 0


def _register_evaluation_worker(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register EvaluationWorkerLoop for GPU nodes to claim gauntlet jobs.

    January 28, 2026: Added to fix gauntlet job claiming.
    Only runs on nodes with gauntlet_enabled=True (i.e., not coordinator).
    Claims evaluation/gauntlet jobs via /work/claim_evaluation endpoint.
    """
    try:
        from app.config.env import env

        # Skip on coordinator nodes - they dispatch but don't execute
        if not env.gauntlet_enabled:
            logger.debug("[LoopRegistry] EvaluationWorkerLoop: skipped (gauntlet_enabled=False)")
            return 0

        from scripts.p2p.loops import (
            EvaluationWorkerLoop,
            EvaluationWorkerConfig,
        )

        async def claim_evaluation_callback(
            node_id: str, capabilities: list[str] | None
        ) -> dict[str, Any] | None:
            """Claim evaluation work from leader via HTTP."""
            leader_url = orchestrator._get_leader_base_url()
            if not leader_url:
                return None
            caps_str = ",".join(capabilities) if capabilities else "gpu,nn,nnue"
            url = f"{leader_url}/work/claim_evaluation?node_id={node_id}&capabilities={caps_str}"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=ClientTimeout(total=30)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("status") == "claimed":
                                return data.get("job")
                        return None
            except Exception as e:
                logger.debug(f"[EvaluationWorkerLoop] Claim error: {e}")
                return None

        async def report_result_callback(job_id: str, results: dict[str, Any]) -> bool:
            """Report evaluation result to leader."""
            leader_url = orchestrator._get_leader_base_url()
            if not leader_url:
                return False
            url = f"{leader_url}/evaluation/result"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json={"job_id": job_id, "results": results},
                        timeout=ClientTimeout(total=30),
                    ) as resp:
                        return resp.status in (200, 201, 204)
            except Exception as e:
                logger.debug(f"[EvaluationWorkerLoop] Report error: {e}")
                return False

        async def report_failure_callback(job_id: str, error: str) -> bool:
            """Report evaluation failure to leader."""
            leader_url = orchestrator._get_leader_base_url()
            if not leader_url:
                return False
            url = f"{leader_url}/evaluation/failure"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json={"job_id": job_id, "error": error},
                        timeout=ClientTimeout(total=30),
                    ) as resp:
                        return resp.status in (200, 201, 204)
            except Exception as e:
                logger.debug(f"[EvaluationWorkerLoop] Failure report error: {e}")
                return False

        eval_worker = EvaluationWorkerLoop(
            node_id=orchestrator.node_id,
            config=EvaluationWorkerConfig(
                interval=30.0,
                claim_timeout=30.0,
                evaluation_timeout=3600.0,
                model_sync_timeout=300.0,
                capabilities=["gpu", "nn", "nnue", "gauntlet"],
            ),
            claim_evaluation_callback=claim_evaluation_callback,
            report_result_callback=report_result_callback,
            report_failure_callback=report_failure_callback,
        )
        manager.register(eval_worker)
        logger.info("[LoopRegistry] EvaluationWorkerLoop registered")
        return 1
    except ImportError as e:
        logger.debug(f"EvaluationWorkerLoop: import failed: {e}")
        failed.append("EvaluationWorkerLoop")
        return 0
    except Exception as e:
        logger.debug(f"EvaluationWorkerLoop: not available: {e}")
        failed.append("EvaluationWorkerLoop")
        return 0


def _register_tournament_data_pipeline(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register TournamentDataPipelineLoop."""
    try:
        from scripts.p2p.loops import (
            TournamentDataPipelineLoop,
            TournamentDataPipelineConfig,
        )

        tournament_pipeline = TournamentDataPipelineLoop(
            get_role=lambda: orchestrator.role,
            config=TournamentDataPipelineConfig(
                interval=3600,
                min_games_for_export=100,
                quality_threshold=0.6,
                emit_training_events=True,
            ),
        )
        manager.register(tournament_pipeline)
        logger.info("[LoopRegistry] TournamentDataPipelineLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"TournamentDataPipelineLoop: not available: {e}")
        failed.append("TournamentDataPipelineLoop")
        return 0


def _register_circuit_breaker_decay(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register CircuitBreakerDecayLoop."""
    try:
        from scripts.p2p.loops.maintenance_loops import (
            CircuitBreakerDecayLoop,
            CircuitBreakerDecayConfig,
        )

        cb_decay_config = CircuitBreakerDecayConfig(
            enabled=os.environ.get("RINGRIFT_CB_DECAY_ENABLED", "1").lower() in ("1", "true", "yes"),
            check_interval_seconds=float(os.environ.get("RINGRIFT_CB_DECAY_INTERVAL", "300")),
            ttl_seconds=float(os.environ.get("RINGRIFT_CB_DECAY_TTL", "3600")),
        )

        cb_decay_loop = CircuitBreakerDecayLoop(config=cb_decay_config)
        cb_decay_loop.set_external_alive_check(orchestrator._is_peer_alive_for_circuit_breaker)
        manager.register(cb_decay_loop)
        orchestrator._cb_decay_loop = cb_decay_loop
        logger.info(f"[LoopRegistry] CircuitBreakerDecayLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"CircuitBreakerDecayLoop: not available: {e}")
        failed.append("CircuitBreakerDecayLoop")
        return 0


def _register_voter_config_sync(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register VoterConfigSyncLoop."""
    try:
        from scripts.p2p.loops.voter_config_sync_loop import VoterConfigSyncLoop

        voter_config_sync = VoterConfigSyncLoop(orchestrator=orchestrator)
        manager.register(voter_config_sync)
        orchestrator._voter_config_sync_loop = voter_config_sync
        logger.info("[LoopRegistry] VoterConfigSyncLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"VoterConfigSyncLoop: not available: {e}")
        failed.append("VoterConfigSyncLoop")
        return 0


def _register_stability_controller(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register StabilityController."""
    try:
        if orchestrator._stability_controller is not None:
            manager.register(orchestrator._stability_controller)
            logger.info("[LoopRegistry] StabilityController registered")
            return 1
        logger.debug("[LoopRegistry] StabilityController: not initialized")
        return 0
    except Exception as e:
        logger.debug(f"StabilityController: registration failed: {e}")
        failed.append("StabilityController")
        return 0


def _register_peer_recovery(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register PeerRecoveryLoop."""
    try:
        from scripts.p2p.loops import PeerRecoveryLoop, PeerRecoveryConfig

        def _get_flapping_peers_for_recovery() -> list[str]:
            if orchestrator._peer_state_tracker:
                return orchestrator._peer_state_tracker.get_flapping_peers()
            return []

        def _get_peer_by_id_for_recovery(node_id: str):
            return orchestrator._peer_snapshot.get_snapshot().get(node_id)

        def _get_retired_peers_for_recovery() -> list:
            return [
                p for p in orchestrator._peer_snapshot.get_snapshot().values()
                if getattr(p, "retired", False) or not p.is_alive()
            ]

        async def _probe_peer_for_recovery(address: str) -> bool:
            try:
                async with orchestrator.http_session.get(
                    f"{address}/health",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    return resp.status == 200
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                return False

        async def _recover_peer_for_recovery(peer) -> bool:
            try:
                node_id = getattr(peer, "node_id", str(peer))
                info = orchestrator.peers.get(node_id)
                if info:
                    info.retired = False
                    info.retired_at = 0.0
                    if orchestrator._cooldown_manager:
                        orchestrator._cooldown_manager.clear_cooldown(node_id)
                    if orchestrator._peer_state_tracker:
                        orchestrator._peer_state_tracker.record_recovery(
                            node_id=node_id,
                            details={"recovery_source": "peer_recovery_loop"},
                        )
                    return True
                return False
            except Exception:
                return False

        def _get_circuit_state_for_recovery(node_id: str) -> str:
            try:
                from app.distributed.circuit_breaker import get_node_circuit_state
                return get_node_circuit_state(node_id) or "CLOSED"
            except Exception:
                return "CLOSED"

        def _reset_circuit_for_recovery(node_id: str) -> bool:
            try:
                from app.distributed.circuit_breaker import reset_node_circuit
                return reset_node_circuit(node_id)
            except Exception:
                return False

        def _get_total_peer_count_for_recovery() -> int:
            return len(orchestrator.peers)

        peer_recovery = PeerRecoveryLoop(
            get_retired_peers=_get_retired_peers_for_recovery,
            probe_peer=_probe_peer_for_recovery,
            recover_peer=_recover_peer_for_recovery,
            emit_event=orchestrator._safe_emit_p2p_event,
            get_circuit_state=_get_circuit_state_for_recovery,
            reset_circuit=_reset_circuit_for_recovery,
            get_total_peer_count=_get_total_peer_count_for_recovery,
            get_flapping_peers=_get_flapping_peers_for_recovery,
            get_peer_by_id=_get_peer_by_id_for_recovery,
            config=PeerRecoveryConfig(),
        )
        manager.register(peer_recovery)
        orchestrator._peer_recovery_loop = peer_recovery
        logger.info("[LoopRegistry] PeerRecoveryLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"PeerRecoveryLoop: not available: {e}")
        failed.append("PeerRecoveryLoop")
        return 0


def _register_git_update(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register GitUpdateLoop.

    January 2026: Replaces inline _git_update_loop in p2p_orchestrator.py (~42 LOC saved).
    """
    try:
        from scripts.p2p.loops.maintenance_loops import GitUpdateLoop, GitUpdateConfig

        git_update = GitUpdateLoop(
            check_for_updates=orchestrator._check_for_updates,
            perform_update=orchestrator._perform_git_update,
            restart_orchestrator=orchestrator._restart_orchestrator,
            get_commits_behind=orchestrator._get_commits_behind,
            config=GitUpdateConfig(),
        )
        manager.register(git_update)
        orchestrator._git_update_loop = git_update
        logger.info("[LoopRegistry] GitUpdateLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"GitUpdateLoop: not available: {e}")
        failed.append("GitUpdateLoop")
        return 0


def _register_voter_heartbeat(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register VoterHeartbeatLoop.

    January 2026: Replaces inline _voter_heartbeat_loop in p2p_orchestrator.py (~101 LOC saved).
    """
    try:
        from scripts.p2p.loops.network_loops import VoterHeartbeatLoop, VoterHeartbeatConfig

        # Cache for voter_id -> peer_key mapping
        _voter_peer_keys: dict[str, str] = {}

        def _get_voter_node_ids() -> list[str]:
            return list(orchestrator.voter_node_ids)

        def _get_node_id() -> str:
            return orchestrator.node_id

        def _get_peer(voter_id: str):
            # Use quorum_manager to find voter peer by IP mapping
            # Cache the peer_key for later use by clear_nat_blocked/increment_failures
            with orchestrator.peers_lock:
                peer_key, peer = orchestrator.quorum_manager.find_voter_peer_by_ip(voter_id)
                if peer_key:
                    _voter_peer_keys[voter_id] = peer_key
            return peer

        async def _send_voter_heartbeat(voter_peer) -> bool:
            return await orchestrator._send_voter_heartbeat(voter_peer)

        async def _try_alternative_endpoints(voter_peer) -> bool:
            return await orchestrator._try_voter_alternative_endpoints(voter_peer)

        async def _discover_voter_peer(voter_id: str) -> None:
            await orchestrator._discover_voter_peer(voter_id)

        async def _refresh_voter_mesh() -> None:
            await orchestrator._refresh_voter_mesh()

        async def _clear_nat_blocked(voter_id: str) -> None:
            # Look up the peer_key from the voter_id (cached during _get_peer)
            peer_key = _voter_peer_keys.get(voter_id)
            if not peer_key:
                # Try to find it again
                with orchestrator.peers_lock:
                    peer_key, _ = orchestrator.quorum_manager.find_voter_peer_by_ip(voter_id)
            if peer_key:
                with orchestrator.peers_lock:
                    if peer_key in orchestrator.peers:
                        orchestrator.peers[peer_key].nat_blocked = False
                        orchestrator.peers[peer_key].nat_blocked_since = 0.0
                        orchestrator.peers[peer_key].consecutive_failures = 0

        def _increment_failures(voter_id: str) -> None:
            # Look up the peer_key from the voter_id (cached during _get_peer)
            peer_key = _voter_peer_keys.get(voter_id)
            if not peer_key:
                # Try to find it again
                with orchestrator.peers_lock:
                    peer_key, _ = orchestrator.quorum_manager.find_voter_peer_by_ip(voter_id)
            if peer_key:
                with orchestrator.peers_lock:
                    if peer_key in orchestrator.peers:
                        orchestrator.peers[peer_key].consecutive_failures = \
                            int(getattr(orchestrator.peers[peer_key], "consecutive_failures", 0) or 0) + 1

        voter_heartbeat = VoterHeartbeatLoop(
            get_voter_node_ids=_get_voter_node_ids,
            get_node_id=_get_node_id,
            get_peer=_get_peer,
            send_voter_heartbeat=_send_voter_heartbeat,
            try_alternative_endpoints=_try_alternative_endpoints,
            discover_voter_peer=_discover_voter_peer,
            refresh_voter_mesh=_refresh_voter_mesh,
            clear_nat_blocked=_clear_nat_blocked,
            increment_failures=_increment_failures,
            config=VoterHeartbeatConfig(),
        )
        manager.register(voter_heartbeat)
        orchestrator._voter_heartbeat_loop = voter_heartbeat
        logger.info("[LoopRegistry] VoterHeartbeatLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"VoterHeartbeatLoop: not available: {e}")
        failed.append("VoterHeartbeatLoop")
        return 0


def _register_reconnect_dead_peers(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register ReconnectDeadPeersLoop.

    January 2026: Replaces inline _reconnect_dead_peers_loop in p2p_orchestrator.py (~138 LOC saved).
    """
    try:
        from scripts.p2p.loops.network_loops import ReconnectDeadPeersLoop, ReconnectDeadPeersConfig
        from scripts.p2p.async_lock import NonBlockingAsyncLockWrapper

        def _get_peers_snapshot() -> list[tuple[str, Any]]:
            """Get a snapshot of all peers."""
            try:
                # Use the async lock in blocking mode for this sync function
                with orchestrator.peers_lock:
                    return list(orchestrator.peers.items())
            except Exception:
                return []

        def _get_node_id() -> str:
            return orchestrator.node_id

        def _get_cached_jittered_timeout() -> float:
            return orchestrator._get_cached_jittered_timeout()

        async def _send_heartbeat_to_peer(host: str, port: int, scheme: str, timeout: float) -> Any:
            return await orchestrator._send_heartbeat_to_peer(host, port, scheme=scheme, timeout=timeout)

        def _get_tailscale_ip_for_peer(node_id: str) -> str | None:
            return orchestrator._get_tailscale_ip_for_peer(node_id)

        def _emit_host_online(node_id: str, capabilities: list[str]) -> None:
            # Jan 30, 2026: Use network orchestrator directly
            try:
                orchestrator.network.emit_host_online_sync(node_id, capabilities)
            except Exception:
                pass

        async def _update_peer_on_success(node_id: str) -> None:
            """Update peer state after successful reconnection."""
            try:
                async with NonBlockingAsyncLockWrapper(orchestrator.peers_lock, "peers_lock", timeout=5.0):
                    if node_id in orchestrator.peers:
                        orchestrator.peers[node_id].consecutive_failures = 0
                        orchestrator.peers[node_id].last_heartbeat = time.time()
            except Exception:
                pass

        reconnect_loop = ReconnectDeadPeersLoop(
            get_peers_snapshot=_get_peers_snapshot,
            get_node_id=_get_node_id,
            get_cached_jittered_timeout=_get_cached_jittered_timeout,
            send_heartbeat_to_peer=_send_heartbeat_to_peer,
            get_tailscale_ip_for_peer=_get_tailscale_ip_for_peer,
            emit_host_online=_emit_host_online,
            update_peer_on_success=_update_peer_on_success,
            config=ReconnectDeadPeersConfig(),
        )
        manager.register(reconnect_loop)
        orchestrator._reconnect_dead_peers_loop = reconnect_loop
        logger.info("[LoopRegistry] ReconnectDeadPeersLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"ReconnectDeadPeersLoop: not available: {e}")
        failed.append("ReconnectDeadPeersLoop")
        return 0


def _register_swim_membership(
    orchestrator: "P2POrchestrator",
    manager: "LoopManager",
    failed: list[str],
) -> int:
    """Register SwimMembershipLoop.

    January 2026: Replaces inline _swim_membership_loop in p2p_orchestrator.py (~92 LOC saved).
    """
    try:
        from scripts.p2p.loops.swim_membership_loop import SwimMembershipLoop, SwimMembershipConfig

        def _get_swim_manager() -> Any:
            return getattr(orchestrator, "_swim_manager", None)

        def _get_peers() -> dict[str, Any]:
            return orchestrator.peers

        def _get_peers_lock() -> Any:
            return orchestrator.peers_lock

        def _try_raft_init() -> None:
            try:
                from scripts.p2p.constants import RAFT_ENABLED
                if (
                    RAFT_ENABLED
                    and not getattr(orchestrator, "_raft_initialized", False)
                    and hasattr(orchestrator, "try_deferred_raft_init")
                ):
                    orchestrator.try_deferred_raft_init()
            except Exception:
                pass

        def _get_raft_initialized() -> bool:
            return getattr(orchestrator, "_raft_initialized", False)

        swim_loop = SwimMembershipLoop(
            get_swim_manager=_get_swim_manager,
            get_peers=_get_peers,
            get_peers_lock=_get_peers_lock,
            config=SwimMembershipConfig(),
            try_raft_init=_try_raft_init,
            get_raft_initialized=_get_raft_initialized,
        )
        manager.register(swim_loop)
        orchestrator._swim_membership_loop = swim_loop
        logger.info("[LoopRegistry] SwimMembershipLoop registered")
        return 1
    except (ImportError, TypeError, AttributeError) as e:
        logger.debug(f"SwimMembershipLoop: not available: {e}")
        failed.append("SwimMembershipLoop")
        return 0
