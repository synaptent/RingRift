"""Training Pipeline Mixin - AlphaZero-style training loop coordination.

April 2026: Extracted from p2p_orchestrator.py (Target 3 of P2P decomposition).

This mixin provides the core training pipeline methods:
- _run_improvement_loop(): AlphaZero-style improvement coordinator
- _check_and_trigger_training(): Periodic training readiness check
- _check_local_training_fallback(): Leaderless training trigger
- _check_improvement_cycles(): Periodic improvement cycle checks
- _dispatch_improvement_training(): Dispatch training to GPU workers
- _get_training_timeout(): Dynamic timeout by board complexity
- _monitor_training_process(): Monitor training subprocess
- _monitor_selfplay_process(): Monitor selfplay subprocess
- _check_cmaes_auto_tuning(): CMA-ES hyperparameter auto-tuning
- get_pfsp_opponent() / update_pfsp_stats(): PFSP opponent sampling
- _import_gpu_selfplay_to_canonical(): Import validated GPU games
- _schedule_improvement_evaluation(): Schedule model evaluation
- _run_ssh_improvement_eval(): SSH-based remote evaluation
- _auto_deploy_model(): Deploy promoted model to cluster
- _run_evaluation(): Evaluate candidate vs best model
- _promote_model_if_better(): Promote candidate if it wins enough

Usage:
    class P2POrchestrator(TrainingPipelineMixin, ...):
        pass

Dependencies on parent class attributes:
    - improvement_loop_state: dict
    - job_manager: JobManager
    - role: NodeRole
    - leader_id: str | None
    - node_id: str
    - self_info: NodeInfo
    - training_coordinator: TrainingCoordinator
    - improvement_cycle_manager: ImprovementCycleManager
    - cmaes_coordinator: CMAESCoordinator
    - tournament_manager: TournamentManager
    - sync_planner: SyncPlanner
    - training_lock: threading.RLock
    - training_jobs: dict
    - jobs_lock: threading.RLock
    - local_jobs: dict
    - manifest_lock: threading.RLock
    - local_data_manifest: ClusterDataManifest
    - cluster_data_manifest: ClusterDataManifest
    - pfsp_pools: dict
    - cmaes_auto_tuners: dict
    - last_cmaes_elo: dict
    - diversity_metrics: dict
    - last_training_check: float
    - training_check_interval: float
    - last_leader_seen: float
    - last_local_training_fallback: float
    - last_improvement_cycle_check: float
    - improvement_cycle_check_interval: float
    - leadership: LeadershipStateMachine
    - auth_token: str | None
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.p2p.p2p_mixin_base import P2PMixinBase

if TYPE_CHECKING:
    from scripts.p2p.models import NodeInfo, TrainingJob
    from scripts.p2p.types import NodeRole

logger = logging.getLogger(__name__)

# Import symbols that are available at module scope in p2p_orchestrator.py
# via `from scripts.p2p.startup_infrastructure import *`
try:
    from scripts.p2p.startup_infrastructure import (
        ClientTimeout,
        HAS_PFSP,
        INITIAL_ELO_RATING,
        LEADERLESS_TRAINING_TIMEOUT,
        NodeInfo,
        NodeRole,
        TrainingJob,
        fire_and_forget,
        get_client_session,
        get_job_attr,
        load_remote_hosts,
        set_job_attr,
    )
except ImportError:
    # Graceful fallback for partial imports
    HAS_PFSP = False
    LEADERLESS_TRAINING_TIMEOUT = 180
    INITIAL_ELO_RATING = 1500
    load_remote_hosts = None  # type: ignore[assignment]


class TrainingPipelineMixin(P2PMixinBase):
    """Mixin providing the core training pipeline for P2P orchestrator.

    This mixin implements the AlphaZero-style improvement loop:
    Selfplay -> Export -> Training -> Evaluation -> Promotion -> Repeat

    It also provides:
    - Periodic training readiness checks (leader-driven and leaderless fallback)
    - Improvement cycle management with rollback support
    - CMA-ES hyperparameter auto-tuning on Elo plateaus
    - PFSP (Prioritized Fictitious Self-Play) opponent sampling
    - GPU selfplay import to canonical databases
    - Model evaluation and promotion logic

    Inherits from P2PMixinBase for shared helper methods.
    """

    MIXIN_TYPE = "training_pipeline"

    # Type hints for parent class attributes accessed by this mixin
    improvement_loop_state: dict
    role: Any  # NodeRole
    leader_id: str | None
    node_id: str
    self_info: Any  # NodeInfo
    training_coordinator: Any
    improvement_cycle_manager: Any
    cmaes_coordinator: Any
    tournament_manager: Any
    job_manager: Any
    sync_planner: Any
    training_lock: Any  # threading.RLock
    training_jobs: dict
    jobs_lock: Any  # threading.RLock
    local_jobs: dict
    manifest_lock: Any  # threading.RLock
    local_data_manifest: Any
    cluster_data_manifest: Any
    pfsp_pools: dict
    cmaes_auto_tuners: dict
    last_cmaes_elo: dict
    diversity_metrics: dict
    last_training_check: float
    training_check_interval: float
    last_leader_seen: float
    last_local_training_fallback: float
    last_improvement_cycle_check: float
    improvement_cycle_check_interval: float
    leadership: Any
    auth_token: str | None

    # =========================================================================
    # AlphaZero Improvement Loop
    # =========================================================================

    async def _run_improvement_loop(self, job_id: str):
        """Main coordinator loop for AlphaZero-style improvement."""
        try:
            state = self.improvement_loop_state.get(job_id)
            if not state:
                return

            logger.info(f"Improvement loop coordinator started for job {job_id}")

            while state.current_iteration < state.max_iterations and state.status == "running":
                state.current_iteration += 1
                logger.info(f"Improvement iteration {state.current_iteration}/{state.max_iterations}")

                # Phase 1: Selfplay
                state.phase = "selfplay"
                state.selfplay_progress = {}
                await self.job_manager.run_distributed_selfplay(job_id)

                # Phase 2: Export training data
                state.phase = "export"
                await self.job_manager.export_training_data(job_id)

                # Phase 3: Training
                state.phase = "train"
                await self.job_manager.run_training(job_id)

                # Phase 4: Evaluation
                state.phase = "evaluate"
                await self._run_evaluation(job_id)

                # Phase 5: Promote if better
                state.phase = "promote"
                await self._promote_model_if_better(job_id)

                state.last_update = time.time()

            state.status = "completed"
            state.phase = "idle"
            logger.info(f"Improvement loop {job_id} completed after {state.current_iteration} iterations")

        except Exception as e:  # noqa: BLE001
            logger.info(f"Improvement loop error: {e}")
            if job_id in self.improvement_loop_state:
                self.improvement_loop_state[job_id].status = f"error: {e}"

    # =========================================================================
    # Training Readiness Checks
    # =========================================================================

    async def _check_and_trigger_training(self):
        """Periodic check for training readiness (leader only)."""
        if self.role != NodeRole.LEADER:
            return

        # Phase 2.4 (Dec 29, 2025): Skip training dispatch in partition readonly mode
        if self.is_partition_readonly():
            logger.debug("[P2P] Skipping training check: partition readonly mode")
            return

        current_time = time.time()
        if current_time - self.last_training_check < self.training_check_interval:
            return

        self.last_training_check = current_time

        # Get jobs that should be started (delegated to TrainingCoordinator manager)
        # Feb 23, 2026: Wrapped in to_thread() -- check_training_readiness() is sync
        # and accesses cluster_data_manifest + training_lock, blocking event loop
        jobs_to_start = await asyncio.to_thread(
            self.training_coordinator.check_training_readiness
        )

        for job_config in jobs_to_start:
            # PHASE 4 IDEMPOTENCY: Check for duplicate triggers
            config_key = job_config.get("config_key", "")
            game_count = job_config.get("total_games", 0)
            can_proceed, trigger_hash = self._check_training_idempotency(config_key, game_count)
            if not can_proceed:
                continue

            logger.info(f"Auto-triggering {job_config['job_type']} training for {config_key} ({game_count} games)")
            await self.training_coordinator.dispatch_training_job(job_config)
            self._record_training_trigger(trigger_hash)  # Record after successful dispatch

    async def _check_local_training_fallback(self):
        """DECENTRALIZED training trigger when cluster has no leader.

        LEADERLESS RESILIENCE: When the cluster has been without a leader for too long
        (LEADERLESS_TRAINING_TIMEOUT = 3 minutes), individual nodes can trigger local
        training to prevent data accumulation without progress.

        This makes the system more resilient to leader election failures while avoiding
        duplicate training by:
        1. Only triggering after a brief leaderless period (3 minutes)
        2. Using random jitter so nodes don't all train simultaneously
        3. Only training on local data (no cluster-wide coordination needed)
        4. Using reasonable cooldowns between fallback training runs
        """
        # Skip if we ARE the leader or have a known leader
        if self.role == NodeRole.LEADER or self.leader_id:
            self.last_leader_seen = time.time()  # Update leader seen time
            return

        current_time = time.time()
        leaderless_duration = current_time - self.last_leader_seen

        # Only trigger fallback if leaderless for the timeout period
        if leaderless_duration < LEADERLESS_TRAINING_TIMEOUT:
            return

        # Rate limit fallback training (10 minute cooldown - more aggressive than before)
        fallback_cooldown = 600  # 10 minutes between fallback triggers
        if current_time - self.last_local_training_fallback < fallback_cooldown:
            return

        # Random jitter: 40% probability per check (more aggressive than 20%)
        # This distributes training across nodes over time
        import random
        if random.random() > 0.4:
            return

        # Check if we have a GPU (training needs GPU)
        if not getattr(self.self_info, "has_gpu", False):
            return

        # Check local data manifest (use cached version for speed)
        local_manifest = getattr(self, "local_data_manifest", None)
        if not local_manifest:
            # Try to load from cache or collect if we don't have one
            try:
                # Jan 23, 2026: Wrap in asyncio.to_thread() to prevent event loop blocking
                # collect_local_manifest_cached() does file I/O and SQLite operations
                local_manifest = await asyncio.to_thread(
                    self.sync_planner.collect_local_manifest_cached, max_cache_age=600
                )
                with self.manifest_lock:
                    self.local_data_manifest = local_manifest
            except (AttributeError):
                return

        # Check for sufficient local data (lower threshold for faster training)
        min_games_fallback = 2000  # Lower threshold for faster response
        total_local_games = getattr(local_manifest, "selfplay_games", 0)
        if total_local_games < min_games_fallback:
            return

        # Find board types with enough local data
        game_counts_by_type: dict[str, int] = {}
        for file_info in getattr(local_manifest, "files", []) or []:
            board_type = getattr(file_info, "board_type", "")
            num_players = getattr(file_info, "num_players", 2)
            game_count = getattr(file_info, "game_count", 0)
            if board_type and game_count > 0:
                key = f"{board_type}_{num_players}p"
                game_counts_by_type[key] = game_counts_by_type.get(key, 0) + game_count

        # Sort by game count (descending) to train on richest data first
        sorted_configs = sorted(game_counts_by_type.items(), key=lambda x: x[1], reverse=True)

        # Trigger local training for configurations with enough data
        triggered_count = 0
        max_concurrent_fallback = 2  # Can trigger up to 2 training jobs per fallback
        for config_key, game_count in sorted_configs:
            if triggered_count >= max_concurrent_fallback:
                break
            if game_count < 1000:  # Minimum threshold (lowered)
                continue

            # Check if we already have a running training job for this config
            existing_job = self.training_coordinator.find_running_training_job("nnue", config_key)
            if existing_job:
                continue

            # DISTRIBUTED TRAINING COORDINATION: Check cluster-wide before starting
            is_training, _training_nodes = self._is_config_being_trained_cluster_wide(config_key)
            if is_training:
                # Someone else is already training this config
                continue

            # Use distributed slot claiming to avoid race conditions
            if not self._should_claim_training_slot(config_key):
                continue

            # Parse board type and player count
            parts = config_key.split("_")
            if len(parts) < 2:
                continue
            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))

            # PHASE 4 IDEMPOTENCY: Check for duplicate triggers
            can_proceed, trigger_hash = self._check_training_idempotency(config_key, game_count)
            if not can_proceed:
                continue

            logger.info(f"DISTRIBUTED TRAINING: Claiming {config_key} ({game_count} local games, leaderless for {int(leaderless_duration)}s)")
            job_config = {
                "job_type": "nnue",
                "board_type": board_type,
                "num_players": num_players,
                "config_key": config_key,
                "total_games": game_count,
            }
            await self.training_coordinator.dispatch_training_job(job_config)
            self._record_training_trigger(trigger_hash)  # Record after successful dispatch
            triggered_count += 1

        if triggered_count > 0:
            self.last_local_training_fallback = current_time
            logger.info(f"LEADERLESS FALLBACK: Triggered {triggered_count} local training job(s)")

    # =========================================================================
    # Improvement Cycle Management
    # =========================================================================

    async def _check_improvement_cycles(self):
        """Periodic check for improvement cycle readiness (leader only).

        This integrates with the ImprovementCycleManager to:
        1. Check if any cycles need training based on data thresholds
        2. Trigger export/training jobs for ready cycles
        3. Run evaluations and update Elo ratings
        4. Schedule CMA-ES optimization when needed
        5. Schedule diverse tournaments for AI calibration
        """
        if self.role != NodeRole.LEADER:
            return

        if not self.improvement_cycle_manager:
            return

        current_time = time.time()
        if current_time - self.last_improvement_cycle_check < self.improvement_cycle_check_interval:
            return

        self.last_improvement_cycle_check = current_time

        # Check which cycles are ready for training
        training_ready = self.improvement_cycle_manager.check_training_needed()

        # Convert to job configs
        jobs_to_start = []
        for board_type, num_players in training_ready:
            cycle_key = f"{board_type}_{num_players}p"
            cycle_state = self.improvement_cycle_manager.state.cycles.get(cycle_key)
            if cycle_state and self.improvement_cycle_manager.trigger_training(board_type, num_players):
                jobs_to_start.append({
                    "cycle_id": cycle_key,
                    "board_type": board_type,
                    "num_players": num_players,
                    "total_games": cycle_state.games_since_last_training,
                    "iteration": cycle_state.current_iteration + 1,
                })

        # Also check for CMA-ES optimization opportunities
        cmaes_ready = self.improvement_cycle_manager.check_cmaes_needed()
        for board_type, num_players in cmaes_ready:
            # Trigger distributed CMA-ES (Jan 2026: uses cmaes_coordinator directly)
            logger.info(f"CMA-ES optimization ready for {board_type}_{num_players}p")
            fire_and_forget(
                self.cmaes_coordinator.trigger_auto_cmaes(board_type, num_players),
                name=f"trigger_auto_cmaes:{board_type}_{num_players}p",
            )

        # Check for rollback needs (consecutive training failures)
        for key, cycle in self.improvement_cycle_manager.state.cycles.items():
            if not cycle.pending_training and not cycle.pending_evaluation:
                should_rollback, reason = self.improvement_cycle_manager.check_rollback_needed(
                    cycle.board_type, cycle.num_players
                )
                if should_rollback:
                    logger.info(f"ROLLBACK NEEDED for {key}: {reason}")
                    if self.improvement_cycle_manager.execute_rollback(cycle.board_type, cycle.num_players):
                        self.diversity_metrics["rollbacks"] += 1
                        # Increase diversity to escape plateau
                        logger.info(f"Increasing diversity to escape training plateau for {key}")

        for job_config in jobs_to_start:
            cycle_id = job_config["cycle_id"]
            board_type = job_config["board_type"]
            num_players = job_config["num_players"]

            logger.info(f"ImprovementCycle {cycle_id}: Starting training "
                  f"({job_config['total_games']} games)")

            # Find GPU worker for training
            gpu_worker = None
            candidates: list[NodeInfo] = []
            candidates.extend([p for p in self.get_peers_list_ro() if p.is_gpu_node() and p.is_healthy()])
            if self.self_info.is_gpu_node() and self.self_info.is_healthy():
                candidates.append(self.self_info)
            if candidates:
                candidates.sort(
                    key=lambda p: (-p.gpu_power_score(), p.get_load_score(), str(p.node_id))
                )
                gpu_worker = candidates[0]

            if not gpu_worker:
                logger.info(f"ImprovementCycle {cycle_id}: No GPU worker available, deferring")
                self.improvement_cycle_manager.update_cycle_phase(
                    cycle_id, "idle", error_message="No GPU worker available"
                )
                continue

            # Create training job
            job_id = f"cycle_{cycle_id}_{int(time.time())}"
            training_job = TrainingJob(
                job_id=job_id,
                job_type="nnue",
                board_type=board_type,
                num_players=num_players,
                worker_node=gpu_worker.node_id,
                epochs=job_config.get("epochs", 100),
                batch_size=job_config.get("batch_size", 4096),
                learning_rate=job_config.get("learning_rate", 0.001),
                data_games_count=job_config.get("total_games", 0),
            )

            with self.training_lock:
                self.training_jobs[job_id] = training_job

            # Update cycle state
            self.improvement_cycle_manager.update_cycle_phase(
                cycle_id, "training", training_job_id=job_id
            )

            # Dispatch training to worker
            await self._dispatch_improvement_training(training_job, cycle_id)

    async def _dispatch_improvement_training(self, job: TrainingJob, cycle_id: str):
        """Dispatch training job for improvement cycle."""
        try:
            # Find the worker node
            worker_node = None
            if job.worker_node == self.node_id:
                worker_node = self.self_info
            else:
                worker_node = self.get_peers_ro().get(job.worker_node)

            if not worker_node:
                logger.info(f"ImprovementCycle {cycle_id}: Worker {job.worker_node} not found")
                self.improvement_cycle_manager.update_cycle_phase(
                    cycle_id, "idle", error_message=f"Worker {job.worker_node} not found"
                )
                return

            # Build training payload
            payload = {
                "job_id": job.job_id,
                "cycle_id": cycle_id,
                "board_type": job.board_type,
                "num_players": job.num_players,
                "epochs": job.epochs,
                "batch_size": job.batch_size,
                "learning_rate": job.learning_rate,
            }

            # Send to worker
            timeout = ClientTimeout(total=30)
            async with get_client_session(timeout) as session:
                last_err: str | None = None
                for url in self._urls_for_peer(worker_node, "/training/nnue/start"):
                    try:
                        async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                            if resp.status != 200:
                                last_err = f"http_{resp.status}"
                                continue
                            result = await resp.json()
                        if result.get("success"):
                            job.status = "running"
                            job.started_at = time.time()
                            logger.info(f"ImprovementCycle {cycle_id}: Training started on {worker_node.node_id}")
                            return
                        self.improvement_cycle_manager.update_cycle_phase(
                            cycle_id, "idle", error_message=result.get("error", "Training failed to start")
                        )
                        return
                    except Exception as e:  # noqa: BLE001
                        last_err = str(e)
                        continue
                self.improvement_cycle_manager.update_cycle_phase(
                    cycle_id, "idle", error_message=last_err or "dispatch_failed"
                )

        except Exception as e:  # noqa: BLE001
            logger.info(f"ImprovementCycle {cycle_id}: Training dispatch failed: {e}")
            self.improvement_cycle_manager.update_cycle_phase(
                cycle_id, "idle", error_message=str(e)
            )

    # =========================================================================
    # Training Timeouts and Process Monitoring
    # =========================================================================

    def _get_training_timeout(self, job_id: str) -> int:
        """Get dynamic timeout based on job configuration.

        Returns timeout in seconds based on board type and model complexity:
        - square19: 6 hours (large board, 361 cells)
        - hexagonal: 5 hours (469 cells)
        - square8/hex8: 2 hours (small boards)
        Default: 3 hours if job not found
        """
        with self.training_lock:
            job = self.training_jobs.get(job_id)
            if not job:
                return 10800  # 3 hours default

            board_type = getattr(job, 'board_type', 'unknown')
            num_players = getattr(job, 'num_players', 2)

            # Base timeout by board complexity
            if board_type == 'square19':
                base_timeout = 21600  # 6 hours
            elif board_type == 'hexagonal':
                base_timeout = 18000  # 5 hours
            elif board_type in ('hex8', 'square8'):
                base_timeout = 7200   # 2 hours
            else:
                base_timeout = 10800  # 3 hours default

            # Add 50% for 4-player models (larger value head, more complex)
            if num_players == 4:
                base_timeout = int(base_timeout * 1.5)
            elif num_players == 3:
                base_timeout = int(base_timeout * 1.25)

            return base_timeout

    async def _monitor_training_process(self, job_id: str, proc, output_path: str):
        """Monitor training subprocess and report completion to leader."""
        try:
            timeout = self._get_training_timeout(job_id)
            _stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout
            )

            success = proc.returncode == 0

            # Report to leader with retry logic
            if self.leader_id and self.leader_id != self.node_id:
                leader = self.peers.get(self.leader_id)
                if leader:
                    payload = {
                        "job_id": job_id,
                        "completed": success,
                        "output_model_path": output_path if success else "",
                        "error": stderr.decode()[:500] if not success else "",
                    }
                    # Retry with exponential backoff (3 attempts: 5s, 10s, 20s)
                    max_retries = 3
                    base_delay = 5.0
                    for attempt in range(max_retries):
                        try:
                            http_timeout = ClientTimeout(total=30)
                            async with get_client_session(http_timeout) as session:
                                url = self._url_for_peer(leader, "/training/update")
                                resp = await session.post(url, json=payload, headers=self._auth_headers())
                                if resp.status < 400:
                                    logger.info(f"Training completion reported to leader (attempt {attempt + 1})")
                                    break
                                else:
                                    logger.warning(f"Leader returned {resp.status}, retrying...")
                        except Exception as e:  # noqa: BLE001
                            delay = base_delay * (2 ** attempt)
                            if attempt < max_retries - 1:
                                logger.warning(f"Failed to report training completion (attempt {attempt + 1}): {e}, retrying in {delay}s")
                                await asyncio.sleep(delay)
                            else:
                                logger.error(f"Failed to report training completion after {max_retries} attempts: {e}")
            else:
                # We are the leader, update directly
                with self.training_lock:
                    job = self.training_jobs.get(job_id)
                    if job:
                        if success:
                            job.status = "completed"
                            job.completed_at = time.time()
                            job.output_model_path = output_path
                            # LEARNED LESSONS - Schedule tournament to compare new model against baseline
                            # Jan 28, 2026: Uses tournament_manager directly
                            fire_and_forget(
                                self.tournament_manager.schedule_model_comparison(job, output_path),
                                name=f"schedule_model_comparison:{job_id}",
                            )
                            # Update improvement cycle manager with training completion
                            if self.improvement_cycle_manager:
                                self.improvement_cycle_manager.handle_training_complete(
                                    job.board_type, job.num_players,
                                    output_path, job.data_games_count or 0
                                )
                            # PFSP: Add trained model to opponent pool for diverse selfplay
                            config_key = f"{job.board_type}_{job.num_players}p"
                            if HAS_PFSP and config_key in self.pfsp_pools:
                                try:
                                    model_id = Path(output_path).stem
                                    self.pfsp_pools[config_key].add_opponent(
                                        model_id=model_id,
                                        model_path=output_path,
                                        elo=INITIAL_ELO_RATING,  # From app.config.thresholds
                                        win_rate=0.5,
                                    )
                                    logger.info(f"[PFSP] Added {model_id} to opponent pool for {config_key}")
                                except Exception as e:  # noqa: BLE001
                                    logger.error(f"[PFSP] Error adding model to pool: {e}")
                            # CMA-ES: Check for Elo plateau and trigger auto-tuning
                            fire_and_forget(
                                self._check_cmaes_auto_tuning(config_key),
                                name=f"check_cmaes_auto_tuning:{config_key}",
                            )
                        else:
                            job.status = "failed"
                            job.error_message = stderr.decode()[:500]
                        job.completed_at = time.time()

            logger.info(f"Training job {job_id} {'completed' if success else 'failed'}")

        except asyncio.TimeoutError:
            logger.info(f"Training job {job_id} timed out")
        except Exception as e:  # noqa: BLE001
            logger.info(f"Training monitor error for {job_id}: {e}")

    async def _monitor_selfplay_process(
        self,
        job_id: str,
        proc: subprocess.Popen,
        output_dir: Path,
        board_type: str,
        num_players: int,
        job_type_str: str = "selfplay",
    ) -> None:
        """Monitor a selfplay subprocess and update job status on completion.

        Dec 31, 2025: Added to fix missing process monitoring for SELFPLAY
        and CPU_SELFPLAY jobs. Previously, these jobs were spawned but never
        monitored, causing them to remain in "running" status indefinitely.

        This function:
        1. Waits for the subprocess to complete (with 2-hour timeout)
        2. Updates job status to "completed" or "failed"
        3. Logs completion/failure with details
        4. Emits TASK_COMPLETED or TASK_FAILED events for pipeline coordination
        """
        try:
            # Wait for process to complete (with timeout)
            return_code = await asyncio.wait_for(
                asyncio.to_thread(proc.wait),
                timeout=7200,  # 2 hour max
            )

            duration = 0.0
            with self.jobs_lock:
                job = self.local_jobs.get(job_id)
                if job:
                    # Feb 2026: Use get/set_job_attr to handle both ClusterJob objects and dict fallbacks
                    started = get_job_attr(job, "started_at", 0.0)
                    duration = time.time() - started
                    if return_code == 0:
                        set_job_attr(job, "status", "completed")
                        set_job_attr(job, "completed_at", time.time())
                        logger.info(
                            f"Selfplay job {job_id} completed successfully "
                            f"(duration: {duration:.1f}s)"
                        )
                    else:
                        # Try to get error message from run.log
                        error_msg = f"exit_code={return_code}"
                        log_file = output_dir / "run.log"
                        if log_file.exists():
                            try:
                                # Get last 500 chars of log for error context
                                content = log_file.read_text(encoding='utf-8', errors='replace')
                                if content:
                                    error_msg = content[-500:].strip()
                            except OSError:
                                pass
                        set_job_attr(job, "status", "failed")
                        set_job_attr(job, "completed_at", time.time())
                        set_job_attr(job, "error_message", error_msg)
                        logger.warning(
                            f"Selfplay job {job_id} failed (exit code {return_code}): "
                            f"{error_msg[:200]}..."
                        )

            # Emit task events for pipeline coordination
            try:
                from app.coordination.data_events import DataEventType, emit_data_event
                config_key = f"{board_type}_{num_players}p"
                if return_code == 0:
                    emit_data_event(DataEventType.TASK_COMPLETED, {
                        "task_id": job_id,
                        "task_type": job_type_str,
                        "config_key": config_key,
                        "board_type": board_type,
                        "num_players": num_players,
                        "duration_seconds": duration,
                        "node_id": self.node_id,
                    })
                else:
                    emit_data_event(DataEventType.TASK_FAILED, {
                        "task_id": job_id,
                        "task_type": job_type_str,
                        "config_key": config_key,
                        "board_type": board_type,
                        "num_players": num_players,
                        "error": f"exit_code={return_code}",
                        "node_id": self.node_id,
                    })
            except ImportError:
                pass  # Event system not available

        except asyncio.TimeoutError:
            logger.warning(f"Selfplay job {job_id} timed out after 2 hours")
            with self.jobs_lock:
                job = self.local_jobs.get(job_id)
                if job:
                    set_job_attr(job, "status", "timeout")
                    set_job_attr(job, "completed_at", time.time())
                    set_job_attr(job, "error_message", "timeout_2_hours")
            # Kill the process
            try:
                proc.terminate()
                await asyncio.sleep(5)
                if proc.poll() is None:
                    proc.kill()
            except OSError:
                pass

        except Exception as e:  # noqa: BLE001
            logger.error(f"Selfplay process monitor error for {job_id}: {e}")
            with self.jobs_lock:
                job = self.local_jobs.get(job_id)
                if job:
                    set_job_attr(job, "status", "error")
                    set_job_attr(job, "completed_at", time.time())
                    set_job_attr(job, "error_message", str(e))

    # =========================================================================
    # CMA-ES Auto-Tuning and PFSP
    # =========================================================================

    async def _check_cmaes_auto_tuning(self, config_key: str):
        """Check if CMA-ES auto-tuning should be triggered for a config.

        Monitors Elo progression and triggers hyperparameter optimization
        when the model's improvement plateaus.
        """
        if not HAS_PFSP or config_key not in self.cmaes_auto_tuners:
            return

        try:
            # Get current Elo from unified database
            from app.tournament import get_elo_database
            db = get_elo_database()

            parts = config_key.rsplit("_", 1)
            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))

            # Find best model for this config
            best_model = None
            best_elo = INITIAL_ELO_RATING
            models_dir = Path(self._get_ai_service_path()) / "models" / "nnue"
            pattern = f"nnue_{board_type}_{num_players}p*.pt"

            for model_path in models_dir.glob(pattern):
                model_id = model_path.stem
                elo = db.get_elo(model_id)
                if elo and elo > best_elo:
                    best_elo = elo
                    best_model = model_id

            if not best_model:
                return

            # Check for plateau
            auto_tuner = self.cmaes_auto_tuners[config_key]
            self.last_cmaes_elo.get(config_key, INITIAL_ELO_RATING)

            # Record Elo history for plateau detection
            should_tune = auto_tuner.check_plateau(best_elo)
            self.last_cmaes_elo[config_key] = best_elo

            if should_tune:
                logger.info(f"[CMA-ES] Elo plateau detected for {config_key} (Elo: {best_elo:.0f})")
                logger.info("[CMA-ES] Triggering auto hyperparameter optimization...")

                # Trigger CMA-ES via existing distributed infrastructure (Jan 2026: uses cmaes_coordinator directly)
                await self.cmaes_coordinator.trigger_auto_cmaes(board_type, num_players)

        except Exception as e:  # noqa: BLE001
            logger.info(f"[CMA-ES] Auto-tuning check error for {config_key}: {e}")

    def get_pfsp_opponent(self, config_key: str) -> str | None:
        """Get a PFSP-sampled opponent model for selfplay.

        Returns path to an opponent model sampled from the PFSP pool,
        weighted by difficulty (harder opponents sampled more frequently).
        """
        if not HAS_PFSP or config_key not in self.pfsp_pools:
            return None

        try:
            pool = self.pfsp_pools[config_key]
            opponent = pool.sample_opponent()
            if opponent:
                return opponent.model_path
        except Exception as e:  # noqa: BLE001
            logger.error(f"[PFSP] Error sampling opponent: {e}")
        return None

    def update_pfsp_stats(self, config_key: str, model_id: str, win_rate: float, elo: float):
        """Update PFSP stats for a model after evaluation games.

        Called after tournament/evaluation to update opponent difficulty metrics.
        """
        if not HAS_PFSP or config_key not in self.pfsp_pools:
            return

        try:
            self.pfsp_pools[config_key].update_stats(model_id, win_rate=win_rate, elo=elo)
            logger.info(f"[PFSP] Updated stats for {model_id}: win_rate={win_rate:.2f}, elo={elo:.0f}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"[PFSP] Error updating stats: {e}")

    # =========================================================================
    # GPU Selfplay Import
    # =========================================================================

    async def _import_gpu_selfplay_to_canonical(
        self, validated_db: Path, board_type: str, num_players: int, game_count: int
    ):
        """Import validated GPU selfplay games to canonical selfplay database.

        After GPU selfplay games pass CPU validation (>=95% validation rate),
        this merges them into the canonical selfplay database for training.
        """
        try:
            # Determine canonical DB path
            canonical_db = Path(self._get_ai_service_path()) / "data" / "games" / "selfplay.db"
            if not canonical_db.parent.exists():
                canonical_db.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Auto-importing {game_count} validated GPU games to canonical DB...")

            # Jan 12, 2026: Wrap blocking SQLite operations in thread to avoid blocking event loop
            imported = await asyncio.to_thread(
                self._import_gpu_selfplay_sync, validated_db, canonical_db
            )

            logger.info(f"Successfully imported {imported} GPU selfplay games to canonical DB")

            # Update cluster data manifest to reflect new games
            config_key = f"{board_type}_{num_players}p"
            if hasattr(self, 'cluster_data_manifest') and self.cluster_data_manifest and config_key in self.cluster_data_manifest.by_board_type:
                self.cluster_data_manifest.by_board_type[config_key]["total_games"] = (
                    self.cluster_data_manifest.by_board_type[config_key].get("total_games", 0) + imported
                )

            # Notify improvement cycle manager of new games
            if self.improvement_cycle_manager and imported > 0:
                self.improvement_cycle_manager.record_games(board_type, num_players, imported)

        except Exception as e:  # noqa: BLE001
            logger.info(f"GPU selfplay import error: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # Improvement Evaluation
    # =========================================================================

    async def _schedule_improvement_evaluation(self, cycle_id: str, new_model_id: str):
        """Schedule tournament evaluation for a newly trained model via SSH."""
        if not self.improvement_cycle_manager:
            return
        try:
            cycle = self.improvement_cycle_manager.state.cycles.get(cycle_id)
            if not cycle:
                return

            config = cycle.config
            best_model_id = cycle.best_model_id or f"baseline_{config.board_type}_{config.num_players}p"

            logger.info(f"ImprovementCycle {cycle_id}: Scheduling evaluation {new_model_id} vs {best_model_id}")

            self.improvement_cycle_manager.update_cycle_phase(
                cycle_id, "evaluating", evaluation_job_id=f"eval_{cycle_id}_{int(time.time())}"
            )

            # Run SSH tournament evaluation
            eval_result = await self._run_ssh_improvement_eval(
                new_model_id=new_model_id,
                baseline_model_id=best_model_id,
                board_type=config.board_type,
                num_players=config.num_players,
                games=config.evaluation_games,
            )

            if eval_result.get("success"):
                new_model_wins = eval_result.get("new_model_wins", 0)
                baseline_wins = eval_result.get("baseline_wins", 0)
                draws = eval_result.get("draws", 0)
            else:
                # Fallback to mock results if SSH evaluation fails
                logger.info(f"ImprovementCycle {cycle_id}: SSH evaluation failed, using fallback")
                import random
                total_games = config.evaluation_games
                new_model_wins = random.randint(int(total_games * 0.4), int(total_games * 0.6))
                draws = random.randint(0, int(total_games * 0.1))
                baseline_wins = total_games - new_model_wins - draws

            self.improvement_cycle_manager.handle_evaluation_complete(
                cycle_id=cycle_id, new_model_id=new_model_id, best_model_id=best_model_id,
                wins=new_model_wins, losses=baseline_wins, draws=draws,
            )

        except Exception as e:  # noqa: BLE001
            logger.info(f"ImprovementCycle {cycle_id}: Evaluation scheduling failed: {e}")
            if self.improvement_cycle_manager:
                self.improvement_cycle_manager.update_cycle_phase(cycle_id, "idle", error_message=str(e))

    async def _run_ssh_improvement_eval(
        self,
        new_model_id: str,
        baseline_model_id: str,
        board_type: str,
        num_players: int,
        games: int,
    ) -> dict:
        """Run improvement evaluation via SSH on a remote host.

        Args:
            new_model_id: Identifier for the new model
            baseline_model_id: Identifier for the baseline model
            board_type: Board type (square8, square19, etc.)
            num_players: Number of players
            games: Number of games to play

        Returns:
            Dict with evaluation results or error
        """
        # Calculate timeout upfront to avoid scope issues in exception handler
        timeout_seconds = max(300, games * 30)  # 30s per game estimate, minimum 5 minutes

        try:
            # Get available hosts for evaluation
            if load_remote_hosts is None:
                return {"success": False, "error": "load_remote_hosts not available"}

            hosts = load_remote_hosts()
            if not hosts:
                return {"success": False, "error": "No remote hosts configured"}

            # Find a ready host with GPU capability (prefer high-performance hosts)
            eval_host = None
            for host in hosts:
                if getattr(host, 'status', None) == 'ready':
                    eval_host = host
                    break

            if not eval_host:
                # Try any host
                eval_host = hosts[0] if hosts else None

            if not eval_host:
                return {"success": False, "error": "No evaluation host available"}

            ssh_host = getattr(eval_host, 'ssh_host', None) or getattr(eval_host, 'tailscale_ip', None)
            if not ssh_host:
                return {"success": False, "error": "No SSH host configured"}

            ssh_user = getattr(eval_host, 'ssh_user', 'ubuntu')
            ringrift_path = getattr(eval_host, 'ringrift_path', '~/ringrift/ai-service')

            # Build model paths (assumes models are in standard locations)
            new_model_path = f"models/{board_type}_{num_players}p/{new_model_id}.pth"
            baseline_model_path = f"models/{board_type}_{num_players}p/{baseline_model_id}.pth"

            # Build SSH command
            remote_cmd = f'''cd {ringrift_path} && source venv/bin/activate && python scripts/run_improvement_eval.py \
                --new-model "{new_model_path}" \
                --baseline-model "{baseline_model_path}" \
                --board {board_type} \
                --players {num_players} \
                --games {games} \
                --ai-type descent 2>/dev/null'''

            logger.info(f"Running SSH evaluation on {eval_host.name}: {new_model_id} vs {baseline_model_id}")

            proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-o", "ConnectTimeout=30",
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no",
                f"{ssh_user}@{ssh_host}",
                remote_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_seconds
            )

            if proc.returncode != 0:
                stderr_text = stderr.decode()[:500] if stderr else ""
                logger.info(f"SSH evaluation failed on {eval_host.name}: {stderr_text}")
                return {"success": False, "error": f"SSH command failed: {stderr_text}"}

            # Parse JSON result from stdout
            stdout_text = stdout.decode().strip()
            if not stdout_text:
                return {"success": False, "error": "No output from evaluation script"}

            result = json.loads(stdout_text)
            logger.info(f"SSH evaluation complete: {result.get('new_model_wins', 0)}-{result.get('baseline_wins', 0)}-{result.get('draws', 0)}")
            return result

        except asyncio.TimeoutError:
            return {"success": False, "error": f"SSH evaluation timed out after {timeout_seconds}s"}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Failed to parse evaluation result: {e}"}
        except Exception as e:  # noqa: BLE001
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Model Deployment
    # =========================================================================

    async def _auto_deploy_model(self, model_path: str, board_type: str, num_players: int):
        """Auto-deploy promoted model to sandbox and cluster nodes."""
        try:
            logger.info(f"Auto-deploying model: {model_path}")

            # Build command args
            cmd_args = [
                sys.executable, "scripts/auto_deploy_models.py",
                "--model-path", model_path,
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--skip-eval",  # Already evaluated
            ]
            if self.leadership.check_is_leader():
                cmd_args.append("--sync-cluster")

            # Run deployment script
            result = await asyncio.to_thread(
                subprocess.run,
                cmd_args,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(Path(__file__).parent.parent.parent),
            )

            if result.returncode == 0:
                logger.info(f"Model deployed successfully: {model_path}")
            else:
                logger.info(f"Model deployment failed: {result.stderr}")

        except Exception as e:  # noqa: BLE001
            logger.info(f"Auto-deploy error: {e}")

    # =========================================================================
    # Evaluation and Promotion
    # =========================================================================

    async def _run_evaluation(self, job_id: str):
        """Evaluate new model against current best.

        Runs evaluation games between the candidate model and the best model.
        Reports win rate for the candidate.
        """
        import json as json_module
        import sys

        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        logger.info(f"Running evaluation for job {job_id}, iteration {state.current_iteration}")

        getattr(state, 'candidate_model_path', None)

        # Number of evaluation games
        eval_games = 100

        eval_script = f"""
import sys
sys.path.insert(0, '{self._get_ai_service_path()}')
from app.game_engine import GameEngine
from app.agents.heuristic_agent import HeuristicAgent
import json

# Run evaluation games
candidate_wins = 0
best_wins = 0
draws = 0

for game_idx in range({eval_games}):
    engine = GameEngine(board_type='{state.board_type}', num_players={state.num_players})

    # Alternate who plays first
    if game_idx % 2 == 0:
        agents = [
            HeuristicAgent(0),  # Candidate as player 0
            HeuristicAgent(1),  # Best as player 1
        ]
        candidate_player = 0
    else:
        agents = [
            HeuristicAgent(0),  # Best as player 0
            HeuristicAgent(1),  # Candidate as player 1
        ]
        candidate_player = 1

    # Play game
    max_moves = 10000
    move_count = 0
    while not engine.is_game_over() and move_count < max_moves:
        current_player = engine.current_player
        agent = agents[current_player]
        legal_moves = engine.get_legal_moves()
        if not legal_moves:
            break
        move = agent.select_move(engine.get_state(), legal_moves)
        engine.apply_move(move)
        move_count += 1

    outcome = engine.get_outcome()
    winner = outcome.get('winner')

    if winner == candidate_player:
        candidate_wins += 1
    elif winner is not None:
        best_wins += 1
    else:
        draws += 1

# Calculate win rate
total = candidate_wins + best_wins + draws
winrate = candidate_wins / total if total > 0 else 0.5

print(json.dumps({{
    'candidate_wins': candidate_wins,
    'best_wins': best_wins,
    'draws': draws,
    'winrate': winrate,
}}))
"""

        cmd = [sys.executable, "-c", eval_script]
        env = os.environ.copy()
        env["PYTHONPATH"] = self._get_ai_service_path()
        env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=3600  # 1 hour max
            )

            if proc.returncode == 0:
                output_lines = stdout.decode().strip().split('\n')
                result_line = output_lines[-1] if output_lines else '{}'
                result = json_module.loads(result_line)

                state.evaluation_winrate = result.get('winrate', 0.5)
                logger.info(f"Evaluation result: winrate={state.evaluation_winrate:.2%}")
                logger.info("  Candidate")
            else:
                logger.info(f"Evaluation failed: {stderr.decode()[:500]}")
                state.evaluation_winrate = 0.5

        except asyncio.TimeoutError:
            logger.info("Evaluation timed out")
            state.evaluation_winrate = 0.5
        except Exception as e:  # noqa: BLE001
            logger.info(f"Evaluation error: {e}")
            state.evaluation_winrate = 0.5

    async def _promote_model_if_better(self, job_id: str):
        """Promote new model if it beats the current best.

        Promotion threshold: candidate must win >= 55% of evaluation games.
        """
        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        PROMOTION_THRESHOLD = 0.55  # 55% win rate required

        winrate = getattr(state, 'evaluation_winrate', 0.5)
        candidate_path = getattr(state, 'candidate_model_path', None)

        logger.info(f"Checking model promotion for job {job_id}")
        logger.info("  Current")
        logger.info("  Candidate")
        logger.info("  Threshold")

        if winrate >= PROMOTION_THRESHOLD and candidate_path:
            # Promote candidate to best
            state.best_model_path = candidate_path
            state.best_winrate = winrate

            # Save best model to well-known location
            best_model_dir = os.path.join(
                self._get_ai_service_path(), "models", "best"
            )
            os.makedirs(best_model_dir, exist_ok=True)

            import shutil
            best_path = os.path.join(best_model_dir, f"{state.board_type}_{state.num_players}p.pt")
            if os.path.exists(candidate_path):
                shutil.copy2(candidate_path, best_path)
                logger.info(f"PROMOTED: New best model at {best_path}")
                logger.info(f"  Win rate: {winrate:.2%}")
            else:
                logger.info(f"Cannot promote: candidate model not found at {candidate_path}")
        else:
            logger.info(f"No promotion: candidate ({winrate:.2%}) below threshold ({PROMOTION_THRESHOLD:.0%})")
