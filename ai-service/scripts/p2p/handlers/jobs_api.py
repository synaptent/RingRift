"""Jobs API HTTP Handlers Mixin.

Provides HTTP endpoints for job management via REST API.
Handles listing, submitting, getting details, and cancelling jobs.

Usage:
    class P2POrchestrator(JobsApiHandlersMixin, ...):
        pass

Endpoints:
    GET /api/jobs - List all jobs with optional filtering
    POST /api/jobs - Submit a new job
    GET /api/jobs/{job_id} - Get job details
    POST /api/jobs/{job_id}/cancel - Cancel a job

Requires the implementing class to have:
    - node_id: str
    - leader_id: str | None
    - jobs_lock: threading.Lock
    - local_jobs: Dict[str, LocalJob]
    - training_lock: threading.Lock
    - training_jobs: Dict[str, TrainingJob]
    - ssh_tournament_lock: threading.Lock
    - ssh_tournament_runs: Dict[str, SSHTournamentRun]
    - training_coordinator: TrainingCoordinator
    - _is_leader() method
    - _proxy_to_leader(request) method
    - _start_local_job(job_type, board_type, num_players, engine_mode) method
    - _save_state() method
    - _emit_task_abandoned(task_id, task_type, reason, node_id) method
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import time
from typing import TYPE_CHECKING, Any

from aiohttp import web

from scripts.p2p.handlers.base import BaseP2PHandler
from scripts.p2p.handlers.timeout_decorator import (
    handler_timeout,
    HANDLER_TIMEOUT_TOURNAMENT,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class JobsApiHandlersMixin(BaseP2PHandler):
    """Mixin providing Jobs API HTTP handlers.

    Inherits from BaseP2PHandler for consistent response formatting.

    Requires the implementing class to have:
    - node_id: str
    - leader_id: str | None
    - jobs_lock: threading.Lock
    - local_jobs: Dict[str, LocalJob]
    - training_lock: threading.Lock
    - training_jobs: Dict[str, TrainingJob]
    - ssh_tournament_lock: threading.Lock
    - ssh_tournament_runs: Dict[str, SSHTournamentRun]
    - training_coordinator: TrainingCoordinator
    - _is_leader() method
    - _proxy_to_leader(request) method
    - _start_local_job(job_type, board_type, num_players, engine_mode) method
    - _save_state() method
    - _emit_task_abandoned(task_id, task_type, reason, node_id) method
    """

    # Type hints for IDE support
    node_id: str
    leader_id: str | None
    jobs_lock: object  # threading.Lock
    local_jobs: dict
    training_lock: object  # threading.Lock
    training_jobs: dict
    ssh_tournament_lock: object  # threading.Lock
    ssh_tournament_runs: dict
    training_coordinator: Any

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_api_jobs_list(self, request: web.Request) -> web.Response:
        """GET /api/jobs - List all jobs with optional filtering.

        Query params:
            type: Filter by job type (e.g., "selfplay", "training", "ssh_tournament")
            status: Filter by status (e.g., "running", "completed", "failed")
            limit: Maximum number of jobs to return (default: 100)

        Returns:
            success: True if operation succeeded
            jobs: List of job objects
            total: Total number of jobs returned
        """
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                proxied = await self._proxy_to_leader(request)
                if proxied.status not in (502, 503):
                    return proxied

            job_type = request.query.get("type")
            status = request.query.get("status")
            limit = int(request.query.get("limit", 100))

            # Collect all jobs (local + training + ssh tournament runs)
            all_jobs = []

            with self.jobs_lock:
                local_jobs_snapshot = list(self.local_jobs.values())
            for job in local_jobs_snapshot:
                jt = job.job_type.value if hasattr(job.job_type, "value") else str(job.job_type)
                if job_type and jt != job_type:
                    continue
                if status and job.status != status:
                    continue
                all_jobs.append(
                    {
                        "job_id": job.job_id,
                        "job_type": jt,
                        "status": job.status,
                        "assigned_to": job.node_id,
                        "created_at": job.started_at,
                        "board_type": job.board_type,
                        "num_players": job.num_players,
                        "category": "local",
                    }
                )

            with self.training_lock:
                for job_id, job in self.training_jobs.items():
                    if job_type and job.job_type != job_type:
                        continue
                    if status and job.status != status:
                        continue
                    all_jobs.append({
                        "job_id": job_id,
                        "job_type": job.job_type,
                        "status": job.status,
                        "assigned_to": job.worker_node,
                        "created_at": job.created_at,
                        "board_type": job.board_type,
                        "num_players": job.num_players,
                        "category": "training",
                    })

            with self.ssh_tournament_lock:
                ssh_runs_snapshot = list(self.ssh_tournament_runs.values())
            for run in ssh_runs_snapshot:
                if job_type and job_type != "ssh_tournament":
                    continue
                if status and run.status != status:
                    continue
                all_jobs.append(
                    {
                        "job_id": run.job_id,
                        "job_type": "ssh_tournament",
                        "status": run.status,
                        "assigned_to": self.node_id,
                        "created_at": run.started_at,
                        "board_type": run.board,
                        "num_players": 2,
                        "category": "ssh_tournament",
                    }
                )

            # Sort by created_at descending and limit
            all_jobs.sort(key=lambda x: x.get("created_at", 0), reverse=True)
            all_jobs = all_jobs[:limit]

            return web.json_response({
                "success": True,
                "jobs": all_jobs,
                "total": len(all_jobs),
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_api_jobs_submit(self, request: web.Request) -> web.Response:
        """POST /api/jobs - Submit a new job via REST API.

        Request body:
            job_type: Type of job ("nnue", "cmaes", "selfplay", "gpu_selfplay", "hybrid_selfplay")
            board_type: Board type (default: "square8")
            num_players: Number of players (default: 2)
            engine_mode: Engine mode for selfplay jobs (default: "gumbel-mcts")
            total_games: Total games for training jobs (default: 0)

        Returns:
            success: True if job was created
            job_id: ID of the created job
            job_type: Type of the job
            status: Initial status
            message: Description message
        """
        if not self._is_leader() and request.query.get("local") != "1":
            return await self._proxy_to_leader(request)
        if not self._is_leader():
            return web.json_response(
                {
                    "success": False,
                    "error": "Not the leader. Please submit to leader node.",
                    "leader_id": self.leader_id,
                },
                status=400,
            )

        try:
            data = await request.json()
            job_type = data.get("job_type")
            if not job_type:
                return web.json_response({
                    "success": False,
                    "error": "job_type is required",
                }, status=400)

            if job_type in ["nnue", "cmaes"]:
                board_type = data.get("board_type", "square8")
                num_players = int(data.get("num_players", 2))
                job_config = {
                    "job_type": job_type,
                    "board_type": board_type,
                    "num_players": num_players,
                    "config_key": f"{board_type}_{num_players}p",
                    "total_games": int(data.get("total_games", 0)),
                }
                job = await self.training_coordinator.dispatch_training_job(job_config)
                if not job:
                    return web.json_response(
                        {"success": False, "error": "No suitable worker available"},
                        status=400,
                    )
                return web.json_response(
                    {
                        "success": True,
                        "job_id": job.job_id,
                        "job_type": job.job_type,
                        "status": job.status,
                        "message": f"Training job {job.job_id} created",
                    }
                )

            if job_type in ["selfplay", "gpu_selfplay", "hybrid_selfplay"]:
                board_type = data.get("board_type", "square8")
                num_players = int(data.get("num_players", 2))
                # Jan 2026: Default to gumbel-mcts for high-quality training data
                engine_mode = data.get("engine_mode", "gumbel-mcts")

                # Map job type string to enum - import lazily
                jt = self._get_job_type_enum(job_type)
                if jt is None:
                    return web.json_response(
                        {"success": False, "error": f"JobType enum not available for: {job_type}"},
                        status=500,
                    )

                job = await self._start_local_job(jt, board_type, num_players, engine_mode)
                if not job:
                    return web.json_response(
                        {"success": False, "error": "Failed to start local job"},
                        status=500,
                    )
                return web.json_response(
                    {
                        "success": True,
                        "job_id": job.job_id,
                        "job_type": job.job_type.value,
                        "status": job.status,
                        "message": f"Job {job.job_id} started",
                    }
                )

            return web.json_response(
                {
                    "success": False,
                    "error": f"Unknown job type: {job_type}. Supported: nnue, cmaes, selfplay, gpu_selfplay, hybrid_selfplay",
                },
                status=400,
            )

        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)}, status=500)

    def _get_job_type_enum(self, job_type_str: str) -> Any:
        """Get JobType enum value from string.

        Handles lazy import to avoid circular dependencies.
        """
        # Try to import JobType from the orchestrator module
        try:
            from scripts.p2p_orchestrator import JobType as JT
            if job_type_str == "gpu_selfplay":
                return JT.GPU_SELFPLAY
            elif job_type_str == "hybrid_selfplay":
                return JT.HYBRID_SELFPLAY
            elif job_type_str == "selfplay":
                return JT.SELFPLAY
            return None
        except ImportError:
            return None

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_api_job_get(self, request: web.Request) -> web.Response:
        """GET /api/jobs/{job_id} - Get details for a specific job.

        Path params:
            job_id: The job ID to retrieve

        Returns:
            success: True if job was found
            job: Job details object
        """
        try:
            job_id = request.match_info.get("job_id")
            if not job_id:
                return web.json_response({
                    "success": False,
                    "error": "job_id is required",
                }, status=400)

            with self.jobs_lock:
                local_job = self.local_jobs.get(job_id)
            if local_job:
                return web.json_response(
                    {
                        "success": True,
                        "job": {
                            "job_id": job_id,
                            "job_type": local_job.job_type.value if hasattr(local_job.job_type, "value") else str(local_job.job_type),
                            "status": local_job.status,
                            "assigned_to": local_job.node_id,
                            "created_at": local_job.started_at,
                            "board_type": local_job.board_type,
                            "num_players": local_job.num_players,
                            "engine_mode": local_job.engine_mode,
                            "pid": local_job.pid,
                            "category": "local",
                        },
                    }
                )

            with self.ssh_tournament_lock:
                ssh_run = self.ssh_tournament_runs.get(job_id)
            if ssh_run:
                return web.json_response(
                    {
                        "success": True,
                        "job": {
                            "job_id": ssh_run.job_id,
                            "job_type": "ssh_tournament",
                            "status": ssh_run.status,
                            "assigned_to": self.node_id,
                            "created_at": ssh_run.started_at,
                            "run_id": ssh_run.run_id,
                            "tiers": ssh_run.tiers,
                            "board_type": ssh_run.board,
                            "games_per_matchup": ssh_run.games_per_matchup,
                            "output_root": ssh_run.output_root,
                            "manifest_path": ssh_run.manifest_path,
                            "checkpoint_path": ssh_run.checkpoint_path,
                            "report_path": ssh_run.report_path,
                            "log_path": ssh_run.log_path,
                            "category": "ssh_tournament",
                        },
                    }
                )

            # Check training jobs
            with self.training_lock:
                if job_id in self.training_jobs:
                    job = self.training_jobs[job_id]
                    return web.json_response({
                        "success": True,
                        "job": {
                            "job_id": job_id,
                            "job_type": job.job_type,
                            "status": job.status,
                            "board_type": job.board_type,
                            "num_players": job.num_players,
                            "assigned_worker": job.worker_node,
                            "created_at": job.created_at,
                            "started_at": job.started_at,
                            "completed_at": job.completed_at,
                            "output_model_path": job.output_model_path,
                            "error_message": job.error_message,
                            "category": "training",
                        },
                    })

            return web.json_response({
                "success": False,
                "error": f"Job {job_id} not found",
            }, status=404)

        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_api_job_cancel(self, request: web.Request) -> web.Response:
        """POST /api/jobs/{job_id}/cancel - Cancel a pending or running job.

        Path params:
            job_id: The job ID to cancel

        Returns:
            success: True if job was cancelled
            message: Description of the result
        """
        if not self._is_leader() and request.query.get("local") != "1":
            return await self._proxy_to_leader(request)
        if not self._is_leader():
            return web.json_response(
                {
                    "success": False,
                    "error": "Not the leader",
                },
                status=400,
            )

        try:
            job_id = request.match_info.get("job_id")
            if not job_id:
                return web.json_response({
                    "success": False,
                    "error": "job_id is required",
                }, status=400)

            with self.jobs_lock:
                local_job = self.local_jobs.get(job_id)
            if local_job:
                with contextlib.suppress(Exception):
                    os.kill(local_job.pid, signal.SIGTERM)
                with self.jobs_lock:
                    local_job.status = "stopped"
                    self.local_jobs[job_id] = local_job
                self._save_state()
                # Emit TASK_ABANDONED event (December 2025)
                await self._emit_task_abandoned(
                    task_id=job_id,
                    task_type=getattr(local_job, "job_type", "local"),
                    reason="user_cancelled",
                    node_id=self.node_id,
                )
                return web.json_response({"success": True, "message": f"Job {job_id} stopped"})

            with self.ssh_tournament_lock:
                ssh_run = self.ssh_tournament_runs.get(job_id)
            if ssh_run:
                if ssh_run.pid:
                    with contextlib.suppress(Exception):
                        os.kill(ssh_run.pid, signal.SIGTERM)
                with self.ssh_tournament_lock:
                    ssh_run.status = "cancelled"
                    ssh_run.completed_at = time.time()
                    self.ssh_tournament_runs[job_id] = ssh_run
                # Emit TASK_ABANDONED event (December 2025)
                await self._emit_task_abandoned(
                    task_id=job_id,
                    task_type="ssh_tournament",
                    reason="user_cancelled",
                    node_id=self.node_id,
                )
                return web.json_response({"success": True, "message": f"SSH tournament {job_id} cancelled"})

            # Check training jobs
            training_cancelled = False
            with self.training_lock:
                if job_id in self.training_jobs:
                    job = self.training_jobs[job_id]
                    if job.status in ["pending", "queued"]:
                        job.status = "cancelled"
                        training_cancelled = True
                    else:
                        return web.json_response({
                            "success": False,
                            "error": f"Cannot cancel job in status: {job.status}",
                        }, status=400)
            if training_cancelled:
                # Emit TASK_ABANDONED event (December 2025)
                await self._emit_task_abandoned(
                    task_id=job_id,
                    task_type="training",
                    reason="user_cancelled",
                    node_id=self.node_id,
                )
                return web.json_response({
                    "success": True,
                    "message": f"Training job {job_id} cancelled",
                })

            return web.json_response({
                "success": False,
                "error": f"Job {job_id} not found",
            }, status=404)

        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)}, status=500)
