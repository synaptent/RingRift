"""Job Management Mixin - cluster/local job orchestration helpers.

April 2026: Extracted from p2p_orchestrator.py (Phase 4 task 16).
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
import uuid
from typing import Any

from scripts.p2p.constants import (
    HTTP_TOTAL_TIMEOUT,
    JOB_CHECK_INTERVAL,
    RELAY_COMMAND_TTL_SECONDS,
    RELAY_MAX_PENDING_START_JOBS,
)
from scripts.p2p.models import ClusterJob, NodeInfo
from scripts.p2p.network import ClientTimeout, get_client_session
from scripts.p2p.p2p_mixin_base import P2PMixinBase
from scripts.p2p.types import JobType, NodeRole

try:
    from app.coordination.safeguards import Safeguards, check_before_spawn
    HAS_SAFEGUARDS = True
    _safeguards = Safeguards.get_instance()
except ImportError:
    HAS_SAFEGUARDS = False
    _safeguards = None

    def check_before_spawn(task_type: str, node_id: str) -> tuple[bool, str]:
        return True, ""

logger = logging.getLogger(__name__)


class JobManagementMixin(P2PMixinBase):
    """Mixin extracted from P2POrchestrator."""

    MIXIN_TYPE = "job_management"

    running: bool
    role: Any
    node_id: str
    self_info: Any
    jobs_lock: Any
    local_jobs: dict[str, Any]
    relay_lock: Any
    relay_command_queue: dict[str, Any]
    peers: dict[str, Any]
    peers_lock: Any
    _job_dispatch_failures: dict[str, Any]
    _JOB_DISPATCH_FAILURE_THRESHOLD: int
    _JOB_DISPATCH_COOLDOWN_SECONDS: float

    async def _run_with_timeout(self, coro, name: str, timeout: float = 60.0) -> None:
        """Run a coroutine with a timeout, logging if it exceeds the limit."""
        try:
            await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"[JobMgmt] {name} timed out after {timeout}s, skipping")
        except Exception as e:
            logger.debug(f"[JobMgmt] {name} failed: {e}")

    async def _run_leader_ops_inline(self) -> None:
        """Run leader-only operations inline in the management loop.

        Feb 23, 2026: Run all operations CONCURRENTLY via asyncio.gather()
        instead of sequentially. Previous sequential execution meant one slow
        op (e.g. manage_cluster_jobs at 361s) blocked ALL other ops and the
        entire event loop. Concurrent execution bounds total cycle time to
        the slowest single op rather than the sum of all ops.

        Individual timeouts via asyncio.wait_for() enforce 15s max per op.
        A hard 60s cycle deadline prevents runaway cycles.
        """
        _t0 = time.time()
        CYCLE_DEADLINE = 60.0  # Hard deadline for entire cycle
        OP_TIMEOUT = 15.0  # Max time per individual operation

        try:
            split_brain = await self._run_with_timeout(
                self._check_and_resolve_split_brain(),
                "check_and_resolve_split_brain", timeout=10.0)
            if split_brain:
                return
            await asyncio.sleep(0)

            logger.info("[LeaderOps] Running leader operations cycle (concurrent)")

            _is_coord = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() in ("true", "1", "yes")
            _leader_ops = [
                ("manage_cluster_jobs", self._manage_cluster_jobs(), OP_TIMEOUT),
                ("check_cluster_balance", self._check_cluster_balance(), OP_TIMEOUT),
                ("check_and_trigger_training", self._check_and_trigger_training(), OP_TIMEOUT),
                ("check_improvement_cycles", self._check_improvement_cycles(), OP_TIMEOUT),
                ("auto_rebalance_from_work_queue",
                 self._auto_rebalance_from_work_queue(), OP_TIMEOUT),
                ("auto_scale_gpu_utilization",
                 self._auto_scale_gpu_utilization(), OP_TIMEOUT),
                ("sweep_nat_recovery",
                 self.recovery_manager.sweep_nat_recovery(), OP_TIMEOUT),
                ("check_node_recovery",
                 self.recovery_manager.check_node_recovery(), OP_TIMEOUT),
            ]
            if not _is_coord:
                _leader_ops.append(("check_and_kill_stuck_jobs",
                    self.job_lifecycle_manager.check_and_kill_stuck_jobs(), OP_TIMEOUT))

            async def _timed_op(name: str, coro, timeout: float) -> None:
                _op_t = time.time()
                await self._run_with_timeout(coro, name, timeout=timeout)
                _op_e = time.time() - _op_t
                if _op_e > 3.0:
                    logger.info(f"[LeaderOps] {name}: {_op_e:.1f}s")

            # Feb 26, 2026: Run ops in batches of 3 to avoid thread pool
            # saturation. With 8 workers and 8 concurrent ops all needing
            # asyncio.to_thread(), the pool gets exhausted and ops queue up
            # causing 22s+ delays. Batching limits peak demand to 3 threads.
            BATCH_SIZE = 3
            try:
                for batch_start in range(0, len(_leader_ops), BATCH_SIZE):
                    if time.time() - _t0 > CYCLE_DEADLINE:
                        logger.warning(f"[LeaderOps] Cycle hit {CYCLE_DEADLINE}s deadline")
                        break
                    batch = _leader_ops[batch_start:batch_start + BATCH_SIZE]
                    remaining = CYCLE_DEADLINE - (time.time() - _t0)
                    await asyncio.wait_for(
                        asyncio.gather(
                            *[_timed_op(n, c, t) for n, c, t in batch],
                            return_exceptions=True,
                        ),
                        timeout=min(remaining, 20.0),
                    )
                    # Yield to event loop between batches
                    await asyncio.sleep(0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"[LeaderOps] Batch hit timeout, "
                    f"cancelling remaining ops"
                )

            _elapsed = time.time() - _t0
            logger.info(f"[LeaderOps] Cycle complete in {_elapsed:.1f}s")
        except Exception as e:
            logger.error(f"[LeaderOps] Error: {e}", exc_info=True)

    async def _job_management_loop(self):
        """Manage jobs - leader coordinates cluster, all nodes handle local operations.

        Feb 22, 2026: Restructured to prevent event loop blocking.
        - Removed redundant gossip ops (already have dedicated LoopManager loops)
        - Skip GPU/selfplay ops on coordinator (no GPU, no selfplay)
        - Reduced timeouts from 60s to 15s
        - Added asyncio.sleep(0) yield points between ops to let HTTP handlers process
        - Run independent leader ops concurrently
        """
        logger.info("[JobMgmt] _job_management_loop started")
        _is_coord = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() in ("true", "1", "yes")
        while self.running:
            try:
                _t0 = time.time()
                # ==== DECENTRALIZED OPERATIONS (all nodes) ====
                # Gossip ops REMOVED - they have dedicated LoopManager loops
                # and were redundantly blocking this loop for 15-30s.
                _ops = [
                    ("check_emergency_coordinator_fallback", lambda: self._check_emergency_coordinator_fallback()),
                ]
                # Feb 2026: Coordinator doesn't run local selfplay/training, so
                # check_local_stuck_jobs (30-54s scanning pgrep/ps) is wasted CPU.
                # Only run on GPU/worker nodes that actually have local jobs.
                if not _is_coord:
                    _ops.append(
                        ("check_local_stuck_jobs", lambda: self.job_lifecycle_manager.check_local_stuck_jobs()),
                    )
                # Coordinator doesn't run selfplay/training locally, doesn't sync
                # from peers, and doesn't need local resource cleanup (no local
                # game data). Skip all heavy ops that use asyncio.to_thread() or
                # subprocess calls - these exhaust the 4-worker thread pool and
                # block HTTP handlers for 15-200s.
                if not _is_coord:
                    _ops.extend([
                        ("local_resource_cleanup", lambda: self.job_coordination_manager.local_resource_cleanup()),
                        ("p2p_data_sync", lambda: self.sync.p2p_data_sync()),
                        ("p2p_model_sync", lambda: self.sync.p2p_model_sync()),
                        ("p2p_training_db_sync", lambda: self.sync.p2p_training_db_sync()),
                    ])
                if not _is_coord:
                    _ops.extend([
                        ("consolidate_selfplay_data", lambda: self.data_pipeline_manager.consolidate_selfplay_data(
                            dispatch_export_job_callback=self._dispatch_export_job)),
                        ("manage_local_jobs_decentralized", lambda: self._manage_local_jobs_decentralized()),
                        ("local_gpu_auto_scale", lambda: self.job_coordination_manager.local_gpu_auto_scale()),
                        ("check_local_training_fallback", lambda: self._check_local_training_fallback()),
                    ])
                for _op_name, _op_factory in _ops:
                    _op_start = time.time()
                    await self._run_with_timeout(_op_factory(), _op_name, timeout=15.0)
                    _op_elapsed = time.time() - _op_start
                    if _op_elapsed > 5.0:
                        logger.warning(f"[JobMgmt] {_op_name} took {_op_elapsed:.1f}s")
                    # Yield to event loop so HTTP handlers can process requests
                    await asyncio.sleep(0)

                # ==== LEADER-ONLY OPERATIONS ====
                # These contain sync blocking calls (subprocess.run, SQLite,
                # check_training_readiness) that block the event loop for
                # 30-136s. Run at reduced frequency (every 60s instead of 15s)
                # and skip during the first cycle to let startup complete.
                if self.role == NodeRole.LEADER:
                    # Mar 6, 2026: Initialize to startup time to skip first cycle.
                    # Previously defaulted to 0, causing time.time()-0 >= 60 = True
                    # immediately, running manage_cluster_jobs (SSH to all nodes)
                    # before event loop stabilized → 59s event loop block.
                    _leader_last = getattr(self, "_last_leader_ops_time", 0)
                    if _leader_last == 0:
                        self._last_leader_ops_time = time.time()
                        _leader_last = self._last_leader_ops_time
                    if time.time() - _leader_last >= 60:
                        self._last_leader_ops_time = time.time()
                        await self._run_leader_ops_inline()
                _total = time.time() - _t0
                logger.info(f"[JobMgmt] Cycle complete in {_total:.1f}s (role={self.role})")
            except Exception as e:  # noqa: BLE001
                logger.error(f"[JobMgmt] Loop error: {e}", exc_info=True)

            await asyncio.sleep(JOB_CHECK_INTERVAL)

    async def _manage_local_jobs_decentralized(self) -> int:
        """DECENTRALIZED: Each node manages its own job count based on gossip state.

        Runs on ALL nodes to ensure selfplay continues even during leader elections.
        Each node autonomously:
        1. Checks its own resource pressure (disk, memory, CPU)
        2. Uses gossip state to calculate proportional job count
        3. Starts or stops local jobs as needed

        PHASE 3 DECENTRALIZATION (Dec 2025):
        - With Serf providing reliable failure detection, we can act quickly
        - Proportional allocation based on gossip cluster capacity
        - 30-second timeout for faster leader-failure recovery

        January 29, 2026: Delegated to ProcessSpawnerOrchestrator.manage_local_jobs_decentralized().

        Returns:
            Number of jobs started/stopped
        """
        # Delegate to ProcessSpawnerOrchestrator if available
        return await self.process_spawner.manage_local_jobs_decentralized()

    async def _auto_scale_gpu_utilization(self) -> int:
        """Auto-scale selfplay jobs to reach 60-80% GPU utilization.

        Detects underutilized GPU nodes and starts selfplay jobs to improve
        cluster throughput while maintaining game quality and rule fidelity.

        Dec 2025 fix: Job type is selected based on GPU capabilities:
        - High-end GPUs (GH200, H100, A100, 5090, 4090): 50% GUMBEL / 50% GPU_SELFPLAY
        - Mid-tier GPUs: HYBRID mode (CPU rules + GPU eval) for rule fidelity

        Returns:
            Number of new selfplay jobs started
        """
        TARGET_GPU_MIN = 60.0  # Target minimum GPU utilization
        TARGET_GPU_MAX = 80.0  # Target maximum GPU utilization
        MIN_IDLE_TIME = 120    # Seconds of low GPU before scaling up

        started = 0
        now = time.time()

        # Rate limit auto-scaling (once per 2 minutes)
        last_scale = getattr(self, "_last_gpu_auto_scale", 0)
        if now - last_scale < 120:
            return 0

        # Feb 23, 2026: Use non-blocking cached snapshot to avoid blocking
        # event loop on peers_lock contention (was 10-30s on macOS)
        peers_snapshot = self._get_peers_snapshot_nonblocking()

        underutilized_gpu_nodes = []

        # Load policy manager for filtering
        policy_manager = None
        try:
            from app.coordination.node_policies import get_policy_manager
            policy_manager = get_policy_manager()
        except ImportError:
            pass

        for peer in peers_snapshot:
            if not peer.is_alive():
                continue
            has_gpu = bool(getattr(peer, "has_gpu", False))
            if not has_gpu:
                continue

            # Policy check: skip nodes that don't allow selfplay
            if policy_manager and not policy_manager.is_work_allowed(peer.node_id, "selfplay"):
                continue

            gpu_percent = float(getattr(peer, "gpu_percent", 0) or 0)
            gpu_name = (getattr(peer, "gpu_name", "") or "").lower()
            selfplay_jobs = int(getattr(peer, "selfplay_jobs", 0) or 0)
            training_jobs = int(getattr(peer, "training_jobs", 0) or 0)

            # Skip if already training
            if training_jobs > 0:
                continue

            # Check if underutilized
            if gpu_percent < TARGET_GPU_MIN:
                # Track how long it's been underutilized
                idle_key = f"_gpu_idle_since_{peer.node_id}"
                idle_since = getattr(self, idle_key, 0)
                if idle_since == 0:
                    setattr(self, idle_key, now)
                elif now - idle_since > MIN_IDLE_TIME:
                    # Calculate how many more jobs to add
                    gpu_headroom = TARGET_GPU_MAX - gpu_percent
                    # Estimate jobs based on GPU tier
                    if any(tag in gpu_name for tag in ("h100", "h200", "gh200", "5090")):
                        jobs_per_10_percent = 2
                    elif any(tag in gpu_name for tag in ("a100", "4090", "3090")):
                        jobs_per_10_percent = 1.5
                    else:
                        jobs_per_10_percent = 1

                    new_jobs = max(1, int(gpu_headroom / 10 * jobs_per_10_percent))
                    new_jobs = min(new_jobs, 4)  # Cap at 4 new jobs per cycle

                    underutilized_gpu_nodes.append({
                        "node_id": peer.node_id,
                        "gpu_percent": gpu_percent,
                        "gpu_name": gpu_name,
                        "current_jobs": selfplay_jobs,
                        "new_jobs": new_jobs,
                    })
            else:
                # GPU is utilized, reset idle timer
                idle_key = f"_gpu_idle_since_{peer.node_id}"
                setattr(self, idle_key, 0)

        # Start GPU selfplay on underutilized nodes
        for node_info in underutilized_gpu_nodes[:3]:  # Max 3 nodes per cycle
            node_id = node_info["node_id"]
            new_jobs = node_info["new_jobs"]

            gpu_name = (node_info.get("gpu_name", "") or "").upper()
            is_high_end = any(tag in gpu_name for tag in ("H100", "H200", "GH200", "A100", "5090", "4090"))
            job_type_str = "GUMBEL/GPU" if is_high_end else "diverse/hybrid"
            print(
                f"[P2P] Auto-scale: {node_id} at {node_info['gpu_percent']:.0f}% GPU, "
                f"starting {new_jobs} {job_type_str} selfplay job(s)"
            )

            for _ in range(new_jobs):
                try:
                    # Schedule selfplay job (Jan 28, 2026: uses job_coordination_manager directly)
                    job = await self.job_coordination_manager.schedule_diverse_selfplay_on_node(node_id)
                    if job:
                        started += 1
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to start diverse selfplay on {node_id}: {e}")
                    break

        if started > 0:
            self._last_gpu_auto_scale = now
            logger.info(f"Auto-scale: started {started} new diverse/hybrid selfplay job(s)")

        return started

    async def _auto_rebalance_from_work_queue(self) -> int:
        """Jan 29, 2026: Delegated to JobOrchestrator.auto_rebalance_from_work_queue()."""
        return await self.jobs.auto_rebalance_from_work_queue()

    async def _check_cluster_balance(self) -> dict[str, Any]:
        """Check and rebalance jobs across the cluster.

        This method identifies:
        1. Powerful nodes that are underutilized (high capacity, low jobs)
        2. Weak nodes that are overloaded (low capacity, high jobs)

        When imbalance is detected, it reduces jobs on weak nodes so the
        scheduler can assign them to more powerful nodes.

        Returns dict with rebalancing actions taken.
        """
        try:
            # Feb 23, 2026: Use non-blocking cached snapshot to avoid blocking
            # event loop on peers_lock contention
            _all_peers = self._get_peers_snapshot_nonblocking()
            alive_peers = [p for p in _all_peers if p.is_alive()]

            all_nodes = [*alive_peers, self.self_info]
            healthy_nodes = [n for n in all_nodes if n.is_healthy()]

            if len(healthy_nodes) < 2:
                return {"action": "none", "reason": "insufficient_nodes"}

            # Calculate capacity and utilization for each node
            node_stats = []
            for node in healthy_nodes:
                target = self.selfplay_scheduler.get_target_jobs_for_node(node)
                current = int(getattr(node, "selfplay_jobs", 0) or 0)
                utilization = current / max(1, target)  # How full is this node
                capacity_score = target  # Higher = more powerful

                node_stats.append({
                    "node": node,
                    "target": target,
                    "current": current,
                    "utilization": utilization,
                    "capacity": capacity_score,
                    "load_score": node.get_load_score(),
                })

            # Find underutilized powerful nodes (capacity > median, utilization < 50%)
            sorted_by_capacity = sorted(node_stats, key=lambda x: x["capacity"], reverse=True)
            median_capacity = sorted_by_capacity[len(sorted_by_capacity) // 2]["capacity"]

            underutilized_powerful = [
                n for n in node_stats
                if n["capacity"] > median_capacity and n["utilization"] < 0.5
            ]

            # Find overloaded weak nodes (capacity < median, utilization > 100%)
            overloaded_weak = [
                n for n in node_stats
                if n["capacity"] < median_capacity and n["utilization"] > 1.0
            ]

            if not underutilized_powerful or not overloaded_weak:
                return {"action": "none", "reason": "balanced"}

            # Calculate rebalancing opportunity
            spare_capacity = sum(
                max(0, n["target"] - n["current"]) for n in underutilized_powerful
            )
            excess_load = sum(
                max(0, n["current"] - n["target"]) for n in overloaded_weak
            )

            if spare_capacity < 2 or excess_load < 2:
                return {"action": "none", "reason": "minimal_imbalance"}

            # Migrate: reduce jobs on weak nodes
            rebalance_actions = []
            jobs_to_migrate = min(spare_capacity, excess_load)

            for weak_node in sorted(overloaded_weak, key=lambda x: x["utilization"], reverse=True):
                if jobs_to_migrate <= 0:
                    break

                node = weak_node["node"]
                reduce_by = min(
                    weak_node["current"] - weak_node["target"],
                    jobs_to_migrate
                )
                new_target = weak_node["current"] - reduce_by

                if reduce_by > 0:
                    print(
                        f"[P2P] Cluster rebalance: {node.node_id} overloaded "
                        f"({weak_node['current']}/{weak_node['target']} jobs, "
                        f"{weak_node['utilization']*100:.0f}% util) - reducing by {reduce_by}"
                    )

                    if node.node_id == self.node_id:
                        await self._reduce_local_selfplay_jobs(new_target, reason="cluster_rebalance")
                    else:
                        await self._request_reduce_selfplay(node, new_target, reason="cluster_rebalance")

                    rebalance_actions.append({
                        "node": node.node_id,
                        "reduced_by": reduce_by,
                        "new_target": new_target,
                    })
                    jobs_to_migrate -= reduce_by

            # Record rebalancing metric
            if rebalance_actions:
                self.record_metric(
                    "cluster_rebalance",
                    len(rebalance_actions),
                    metadata={
                        "spare_capacity": spare_capacity,
                        "excess_load": excess_load,
                        "actions": rebalance_actions,
                    },
                )

            return {
                "action": "rebalanced",
                "spare_capacity": spare_capacity,
                "excess_load": excess_load,
                "actions": rebalance_actions,
            }

        except Exception as e:  # noqa: BLE001
            logger.info(f"Cluster balance check error: {e}")
            return {"action": "error", "error": str(e)}

    async def _manage_cluster_jobs(self):
        """Manage jobs across the cluster (leader only).

        Jan 29, 2026: Delegated to ProcessSpawnerOrchestrator.
        LEARNED LESSONS incorporated:
        - Check disk space BEFORE starting jobs (Vast.ai 91-93% disk issue)
        - Check memory to prevent OOM (AWS instance crashed at 31GB+)
        - Trigger cleanup when approaching limits
        - Use is_healthy() not just is_alive()
        """
        # Jan 29, 2026: Delegate to ProcessSpawnerOrchestrator
        return await self.process_spawner.manage_cluster_jobs()

    async def _cleanup_local_disk(self):
        """Clean up disk space on local node.

        Jan 2026: Delegated to MemoryDiskManager (Phase 10 decomposition).
        """
        await self.memory_disk_manager.cleanup_local_disk()

    async def _request_remote_cleanup(self, node: NodeInfo):
        """Request a remote node to clean up disk space.

        Jan 2026: Delegated to MemoryDiskManager (Phase 10 decomposition).
        """
        await self.memory_disk_manager.request_remote_cleanup_via_orchestrator(node)

    async def _reduce_local_selfplay_jobs(self, target_selfplay_jobs: int, *, reason: str) -> dict[str, Any]:
        """Best-effort: stop excess selfplay jobs on this node (load shedding).

        Jan 2026: Delegated to MemoryDiskManager (Phase 10 decomposition).
        """
        return await self.memory_disk_manager.reduce_local_selfplay_jobs(target_selfplay_jobs, reason=reason)

    async def _request_reduce_selfplay(self, node: NodeInfo, target_selfplay_jobs: int, *, reason: str) -> None:
        """Ask a node to shed excess selfplay (used for memory/disk pressure).

        Jan 2026: Delegated to MemoryDiskManager (Phase 10 decomposition).
        """
        await self.memory_disk_manager.request_reduce_selfplay(node, target_selfplay_jobs, reason=reason)

    async def _restart_local_stuck_jobs(self):
        """Kill stuck selfplay processes and let job management restart them.

        LEARNED LESSONS - Addresses the issue where processes accumulate but GPU stays at 0%.
        """
        logger.info("Restarting stuck local selfplay jobs...")
        try:
            # Kill tracked selfplay jobs (avoid broad pkill patterns).
            jobs_to_clear: list[str] = []
            pids_to_kill: set[int] = set()
            with self.jobs_lock:
                for job_id, job in self.local_jobs.items():
                    if job.job_type not in (JobType.SELFPLAY, JobType.GPU_SELFPLAY, JobType.HYBRID_SELFPLAY, JobType.CPU_SELFPLAY, JobType.GUMBEL_SELFPLAY):
                        continue
                    jobs_to_clear.append(job_id)
                    if job.pid:
                        try:
                            pids_to_kill.add(int(job.pid))
                        except (ValueError, AttributeError):
                            continue

            # Sweep for untracked selfplay processes (e.g. lost local_jobs state) and kill them too.
            try:
                import shutil

                if shutil.which("pgrep"):
                    # December 2025: Added selfplay.py - unified entry point
                    for pattern in (
                        "selfplay.py",
                        "run_self_play_soak.py",
                        "run_gpu_selfplay.py",
                        "run_hybrid_selfplay.py",
                        "run_random_selfplay.py",
                    ):
                        out = subprocess.run(
                            ["pgrep", "-f", pattern],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if out.returncode == 0 and out.stdout.strip():
                            for token in out.stdout.strip().split():
                                try:
                                    pids_to_kill.add(int(token))
                                except (ValueError, AttributeError):
                                    continue
            except (ValueError, AttributeError):
                pass

            pids_to_kill.discard(int(os.getpid()))

            killed = 0
            for pid in sorted(pids_to_kill):
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed += 1
                except (AttributeError):
                    continue

            # Clear our job tracking - they'll be restarted next cycle.
            with self.jobs_lock:
                for job_id in jobs_to_clear:
                    self.local_jobs.pop(job_id, None)

            logger.info(f"Killed {killed} processes, cleared {len(jobs_to_clear)} job records")
        except Exception as e:  # noqa: BLE001
            logger.error(f"killing stuck processes: {e}")

    async def _request_job_restart(self, node: NodeInfo):
        """Request a remote node to restart its stuck selfplay jobs."""
        try:
            if getattr(node, "nat_blocked", False):
                cmd_id = await self._enqueue_relay_command_for_peer(node, "restart_stuck_jobs", {})
                if cmd_id:
                    logger.info(f"Enqueued relay restart_stuck_jobs for {node.node_id}")
                else:
                    logger.info(f"Relay queue full for {node.node_id}; skipping restart enqueue")
                return
            timeout = ClientTimeout(total=HTTP_TOTAL_TIMEOUT)
            async with get_client_session(timeout) as session:
                last_err: str | None = None
                for url in self._urls_for_peer(node, "/restart_stuck_jobs"):
                    try:
                        async with session.post(url, json={}, headers=self._auth_headers()) as resp:
                            if resp.status != 200:
                                last_err = f"http_{resp.status}"
                                continue
                            data = await resp.json()
                            if data.get("success"):
                                logger.info(f"Job restart requested on {node.node_id}")
                                return
                            last_err = str(data.get("error") or "restart_failed")
                    except Exception as e:  # noqa: BLE001
                        last_err = str(e)
                        continue
                if last_err:
                    logger.info(f"Job restart request failed on {node.node_id}: {last_err}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to request job restart from {node.node_id}: {e}")

    async def _start_local_job(
        self,
        job_type: JobType,
        board_type: str = "square8",
        num_players: int = 2,
        engine_mode: str = "gumbel-mcts",  # GPU-accelerated Gumbel MCTS
        job_id: str | None = None,
        cuda_visible_devices: str | None = None,
        export_params: dict[str, Any] | None = None,
        simulation_budget: int | None = None,  # Gumbel MCTS budget (None = use tier default)
    ) -> ClusterJob | None:
        """Start a job on the local node.

        Jan 29, 2026: Delegated to ProcessSpawnerOrchestrator.
        SAFEGUARD: Checks coordination safeguards before spawning.
        """
        # Jan 29, 2026: Delegate to ProcessSpawnerOrchestrator
        return await self.process_spawner.start_local_job(
            job_type=job_type,
            board_type=board_type,
            num_players=num_players,
            engine_mode=engine_mode,
            job_id=job_id,
            cuda_visible_devices=cuda_visible_devices,
            export_params=export_params,
            simulation_budget=simulation_budget,
        )

    async def _dispatch_export_job(
        self,
        node: NodeInfo,
        input_path: str,
        output_path: str,
        board_type: str,
        num_players: int,
        encoder_version: str = "v3",
        max_games: int = 5000,
        is_jsonl: bool = False,
    ):
        """Dispatch a CPU-intensive export job to a high-CPU node.

        CPU-intensive jobs like NPZ export should run on vast nodes
        (256-512 CPUs) rather than lambda nodes (64 CPUs) to free
        GPU resources for training/selfplay.
        """
        try:
            job_id = f"export_{board_type}_{num_players}p_{int(time.time())}_{uuid.uuid4().hex[:6]}"

            payload = {
                "job_id": job_id,
                "job_type": JobType.DATA_EXPORT.value,
                "board_type": board_type,
                "num_players": num_players,
                "input_path": input_path,
                "output_path": output_path,
                "encoder_version": encoder_version,
                "max_games": max_games,
                "is_jsonl": is_jsonl,
            }

            # NAT-blocked nodes need relay command
            if getattr(node, "nat_blocked", False):
                cmd_id = await self._enqueue_relay_command_for_peer(node, "start_job", payload)
                if cmd_id:
                    logger.info(f"Enqueued relay export job for {node.node_id}: {job_id}")
                else:
                    logger.info(f"Relay queue full for {node.node_id}; export not dispatched")
                return

            timeout = ClientTimeout(total=30)
            async with get_client_session(timeout) as session:
                last_err: str | None = None
                for url in self._urls_for_peer(node, "/start_job"):
                    try:
                        async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                            if resp.status == 200:
                                result = await resp.json()
                                if result.get("success"):
                                    logger.info(f"Export job dispatched to {node.node_id}: {job_id}")
                                    return
                                last_err = result.get("error", "unknown")
                            else:
                                last_err = f"http_{resp.status}"
                    except Exception as e:  # noqa: BLE001
                        last_err = str(e)

                if last_err:
                    logger.info(f"Export job dispatch failed to {node.node_id}: {last_err}")

        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to dispatch export job to {node.node_id}: {e}")

    async def _request_remote_job(
        self,
        node: NodeInfo,
        job_type: JobType,
        board_type: str = "square8",
        num_players: int = 2,
        engine_mode: str = "hybrid",
    ):
        """Request a remote node to start a job with specific configuration.

        SAFEGUARD: Checks coordination safeguards before requesting remote spawn.
        """
        try:
            # Feb 2026: Normalize job_type to string early — _get_job_type() returns
            # a plain string when the JobType enum isn't available, but callers below
            # assumed .value always exists. This caused 4000+ dispatch failures/day.
            job_type_str = job_type.value if hasattr(job_type, 'value') else str(job_type)

            # Feb 2026: Check per-node dispatch cooldown to prevent tight retry loops.
            # Nodes that fail repeatedly get skipped for _JOB_DISPATCH_COOLDOWN_SECONDS.
            nid = node.node_id
            fail_info = self._job_dispatch_failures.get(nid)
            if fail_info:
                fail_count, fail_time = fail_info
                if fail_count >= self._JOB_DISPATCH_FAILURE_THRESHOLD:
                    elapsed = time.time() - fail_time
                    if elapsed < self._JOB_DISPATCH_COOLDOWN_SECONDS:
                        return  # Silently skip — already logged when cooldown started
                    else:
                        # Cooldown expired, reset and allow retry
                        self._job_dispatch_failures[nid] = (0, 0.0)

            # SAFEGUARD: Check safeguards before requesting remote spawn
            if HAS_SAFEGUARDS and _safeguards:
                allowed, reason = check_before_spawn(job_type_str, node.node_id)
                if not allowed:
                    logger.info(f"SAFEGUARD blocked remote {job_type_str} on {node.node_id}: {reason}")
                    return

            job_id = f"{job_type_str}_{board_type}_{num_players}p_{int(time.time())}_{uuid.uuid4().hex[:6]}"

            # NAT-blocked nodes can't accept inbound /start_job; enqueue a relay command instead.
            if getattr(node, "nat_blocked", False):
                payload = {
                    "job_id": job_id,
                    "job_type": job_type_str,
                    "board_type": board_type,
                    "num_players": num_players,
                    "engine_mode": engine_mode,
                }
                cmd_id = await self._enqueue_relay_command_for_peer(node, "start_job", payload)
                if cmd_id:
                    print(
                        f"[P2P] Enqueued relay job for {node.node_id}: "
                        f"{job_type_str} {board_type} {num_players}p ({job_id})"
                    )
                else:
                    logger.info(f"Relay queue full for {node.node_id}; skipping enqueue")
                return

            timeout = ClientTimeout(total=10)
            async with get_client_session(timeout) as session:
                payload = {
                    "job_id": job_id,
                    "job_type": job_type_str,
                    "board_type": board_type,
                    "num_players": num_players,
                    "engine_mode": engine_mode,
                }
                last_err: str | None = None
                for url in self._urls_for_peer(node, "/start_job"):
                    try:
                        async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                            if resp.status != 200:
                                last_err = f"http_{resp.status}"
                                continue
                            data = await resp.json()
                            if data.get("success"):
                                logger.info(f"Started remote {board_type} {num_players}p job on {node.node_id}")
                                # Reset failure tracking on success
                                self._job_dispatch_failures.pop(nid, None)
                                return
                            last_err = str(data.get("error") or "start_failed")
                    except Exception as e:  # noqa: BLE001
                        last_err = str(e)
                        continue
                if last_err:
                    # Track consecutive failures for cooldown
                    prev_count = self._job_dispatch_failures.get(nid, (0, 0.0))[0]
                    new_count = prev_count + 1
                    self._job_dispatch_failures[nid] = (new_count, time.time())
                    if new_count >= self._JOB_DISPATCH_FAILURE_THRESHOLD:
                        logger.warning(
                            f"Job dispatch to {nid} failed {new_count}x consecutively, "
                            f"cooling down for {self._JOB_DISPATCH_COOLDOWN_SECONDS}s"
                        )
                    else:
                        logger.error(f"Failed to start remote job on {nid}: {last_err}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to request remote job from {node.node_id}: {e}")

    def _enqueue_relay_command(self, node_id: str, cmd_type: str, payload: dict[str, Any]) -> str | None:
        """Leader-side: enqueue a command for a NAT-blocked node to pull."""
        now = time.time()
        cmd_type = str(cmd_type)
        payload = dict(payload or {})

        with self.relay_lock:
            queue = list(self.relay_command_queue.get(node_id, []))
            queue = [
                cmd for cmd in queue
                if float(cmd.get("expires_at", 0.0) or 0.0) > now
            ]

            if cmd_type == "start_job":
                pending = sum(1 for c in queue if str(c.get("type") or "") == "start_job")
                if pending >= RELAY_MAX_PENDING_START_JOBS:
                    self.relay_command_queue[node_id] = queue
                    return None

                job_id = str(payload.get("job_id") or "")
                if job_id:
                    for c in queue:
                        if str(c.get("payload", {}).get("job_id") or "") == job_id:
                            self.relay_command_queue[node_id] = queue
                            return str(c.get("id") or "")

            cmd_id = uuid.uuid4().hex
            queue.append(
                {
                    "id": cmd_id,
                    "type": cmd_type,
                    "payload": payload,
                    "created_at": now,
                    "expires_at": now + RELAY_COMMAND_TTL_SECONDS,
                }
            )
            self.relay_command_queue[node_id] = queue
            return cmd_id

    async def _enqueue_relay_command_for_peer(
        self,
        peer: NodeInfo,
        cmd_type: str,
        payload: dict[str, Any],
    ) -> str | None:
        """Enqueue a relay command for `peer`, forwarding via its relay hub when needed.

        Default behavior: NAT-blocked nodes poll the leader's `/relay/heartbeat`
        endpoint and the leader stores commands in-memory.

        Some nodes (notably certain containerized GPU providers) may be unable to
        reach the leader over the mesh network (e.g. TUN-less Tailscale) and also
        cannot accept inbound connections. Those nodes will instead send relay
        heartbeats to an internet-reachable hub (e.g. `aws-staging`). When
        `peer.relay_via` points to such a hub, the leader must enqueue the relay
        command on that hub so the node can pull and execute it.
        """
        if not peer or not getattr(peer, "node_id", ""):
            return None

        peer_id = str(getattr(peer, "node_id", "") or "").strip()
        if not peer_id:
            return None

        relay_node_id = str(getattr(peer, "relay_via", "") or "").strip()
        if relay_node_id and relay_node_id != self.node_id:
            relay_peer = self.get_peers_ro().get(relay_node_id)
            if relay_peer:
                timeout = ClientTimeout(total=10)
                async with get_client_session(timeout) as session:
                    last_err: str | None = None
                    for url in self._urls_for_peer(relay_peer, "/relay/enqueue"):
                        try:
                            async with session.post(
                                url,
                                json={
                                    "target_node_id": peer_id,
                                    "type": cmd_type,
                                    "payload": payload or {},
                                },
                                headers=self._auth_headers(),
                            ) as resp:
                                if resp.status != 200:
                                    last_err = f"http_{resp.status}"
                                    continue
                                data = await resp.json()
                                if data.get("success"):
                                    return str(data.get("id") or "")
                                last_err = str(data.get("error") or "enqueue_failed")
                        except Exception as e:  # noqa: BLE001
                            last_err = str(e)
                            continue
                    if last_err:
                        logger.info(f"Relay enqueue via {relay_node_id} failed for {peer_id}: {last_err}")
                        # Dec 30, 2025: Automatic relay failover
                        # If the current relay is unreachable, try to find a new one
                        # January 4, 2026: Pass peer_id for configured relay preferences
                        # Inline: was _select_best_relay()
                        new_relay = self.recovery_manager.select_best_relay(for_peer=peer_id)
                        if new_relay and new_relay != relay_node_id:
                            logger.info(
                                f"[RelayFailover] Switching {peer_id} relay: "
                                f"{relay_node_id} -> {new_relay}"
                            )
                            with self.peers_lock:
                                if peer_id in self.peers:
                                    self.peers[peer_id].relay_via = new_relay
                                    self._publish_peers_snapshot()
                            # Try enqueue on new relay
                            new_relay_peer = self.get_peers_ro().get(new_relay)
                            if new_relay_peer:
                                for url in self._urls_for_peer(new_relay_peer, "/relay/enqueue"):
                                    try:
                                        timeout = ClientTimeout(total=10)
                                        async with get_client_session(timeout) as session2:
                                            async with session2.post(
                                                url,
                                                json={
                                                    "target_node_id": peer_id,
                                                    "type": cmd_type,
                                                    "payload": payload or {},
                                                },
                                                headers=self._auth_headers(),
                                            ) as resp2:
                                                if resp2.status == 200:
                                                    data2 = await resp2.json()
                                                    if data2.get("success"):
                                                        return str(data2.get("id") or "")
                                    except Exception:  # noqa: BLE001
                                        continue

        # Fallback: enqueue locally (works when peer polls the leader directly).
        return self._enqueue_relay_command(peer_id, cmd_type, payload)
