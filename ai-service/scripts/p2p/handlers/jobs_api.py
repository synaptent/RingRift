"""Jobs API HTTP Handlers Mixin.

Provides HTTP endpoints for job management via REST API.
Handles listing, submitting, getting details, cancelling, and managing jobs.

Usage:
    class P2POrchestrator(JobsApiHandlersMixin, ...):
        pass

Endpoints:
    GET /api/jobs - List all jobs with optional filtering
    POST /api/jobs - Submit a new job
    GET /api/jobs/{job_id} - Get job details
    POST /api/jobs/{job_id}/cancel - Cancel a job
    POST /start_job - Start a job (from leader)
    POST /stop_job - Stop a running job
    POST /job/kill - Forcefully kill a stuck job
    POST /cleanup - Trigger disk cleanup
    POST /restart_stuck_jobs - Restart stuck selfplay jobs
    POST /cleanup_files - Delete specific files

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
    - _cleanup_local_disk() method
    - _get_resource_usage() method
    - _restart_local_stuck_jobs() method
    - get_data_directory() method
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

from app.core.async_context import safe_create_task
from scripts.p2p.network import NonBlockingAsyncLockWrapper

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
        Feb 2026: Use JobType(value) directly instead of hardcoded if-chain.
        The old code only handled 3 of 12+ job types, silently returning None
        for gumbel_selfplay, cpu_selfplay, training, etc.
        """
        # Try local types module first (no circular dependency risk)
        try:
            from scripts.p2p.types import JobType as JT
            return JT(job_type_str)
        except (ImportError, ValueError):
            pass
        # Fallback to orchestrator module
        try:
            from scripts.p2p_orchestrator import JobType as JT
            return JT(job_type_str)
        except (ImportError, ValueError):
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

    # -------------------------------------------------------------------------
    # Job Management Handlers - Extracted from p2p_orchestrator.py
    # January 2026 - P2P Modularization Phase 7a
    # -------------------------------------------------------------------------

    async def handle_start_job(self, request: web.Request) -> web.Response:
        """Handle request to start a job (from leader).

        Request body:
            job_type: Type of job (selfplay, gpu_selfplay, data_export, etc.)
            board_type: Board type (default: square8)
            num_players: Number of players (default: 2)
            engine_mode: Engine mode (default: gumbel-mcts)
            job_id: Optional job ID
            cuda_visible_devices: Optional CUDA device selection
            export_params: Optional params for DATA_EXPORT jobs

        Returns:
            success: True if job started
            job: Job details dict
        """
        try:
            data = await request.json()

            # Import JobType lazily to avoid circular imports
            job_type_enum = self._get_job_type_enum(data.get("job_type", "selfplay"))
            if job_type_enum is None:
                # Try direct import for all job types
                try:
                    from scripts.p2p_orchestrator import JobType
                    job_type_enum = JobType(data.get("job_type", "selfplay"))
                except (ImportError, ValueError):
                    return web.json_response(
                        {"error": f"Unknown job type: {data.get('job_type')}"},
                        status=400,
                    )

            board_type = data.get("board_type", "square8")
            num_players = data.get("num_players", 2)
            engine_mode = data.get("engine_mode", "gumbel-mcts")
            job_id = data.get("job_id")
            cuda_visible_devices = data.get("cuda_visible_devices")

            # Extra params for DATA_EXPORT jobs
            export_params = None
            try:
                from scripts.p2p_orchestrator import JobType as JT
                if job_type_enum == JT.DATA_EXPORT:
                    export_params = {
                        "input_path": data.get("input_path"),
                        "output_path": data.get("output_path"),
                        "encoder_version": data.get("encoder_version", "v3"),
                        "max_games": data.get("max_games", 5000),
                        "is_jsonl": data.get("is_jsonl", False),
                    }
            except ImportError:
                pass

            job = await self._start_local_job(
                job_type_enum,
                board_type=board_type,
                num_players=num_players,
                engine_mode=engine_mode,
                job_id=job_id,
                cuda_visible_devices=cuda_visible_devices,
                export_params=export_params,
            )

            if job:
                return web.json_response({"success": True, "job": job.to_dict()})
            else:
                return web.json_response(
                    {"success": False, "error": "Failed to start job"},
                    status=500,
                )
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=400)

    async def handle_stop_job(self, request: web.Request) -> web.Response:
        """Handle request to stop a job.

        Request body:
            job_id: ID of the job to stop

        Returns:
            success: True if job was stopped
        """
        try:
            data = await request.json()
            job_id = data.get("job_id")

            async with NonBlockingAsyncLockWrapper(self.jobs_lock, "jobs_lock", timeout=5.0):
                if job_id in self.local_jobs:
                    job = self.local_jobs[job_id]
                    try:
                        os.kill(job.pid, signal.SIGTERM)
                        job.status = "stopped"
                    except OSError:
                        pass  # Process already dead
                    return web.json_response({"success": True})

            return web.json_response(
                {"success": False, "error": "Job not found"},
                status=404,
            )
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=400)

    async def handle_job_kill(self, request: web.Request) -> web.Response:
        """Handle request to forcefully kill a stuck job (SIGKILL).

        Used by the leader's self-healing system to kill stuck jobs remotely.
        Supports killing by job_id or by job_type pattern.

        Request body:
            job_id: Optional job ID to kill
            job_type: Optional job type pattern ("training", "selfplay", etc.)
            reason: Reason for killing the job

        Returns:
            success: True if any processes were killed
            killed: Number of processes killed
            reason: The provided reason
        """
        try:
            data = await request.json()
            job_id = data.get("job_id")
            job_type = data.get("job_type")  # "training", "selfplay", etc.
            reason = data.get("reason", "unknown")

            killed = 0

            # Try to kill by job_id first
            if job_id:
                async with NonBlockingAsyncLockWrapper(self.jobs_lock, "jobs_lock", timeout=5.0):
                    if job_id in self.local_jobs:
                        job = self.local_jobs[job_id]
                        try:
                            os.kill(job.pid, signal.SIGKILL)
                            job.status = "killed"
                            killed += 1
                            logger.info(f"Killed job {job_id} (pid {job.pid}): {reason}")
                        except Exception as e:  # noqa: BLE001
                            logger.error(f"Failed to kill job {job_id}: {e}")

            # Kill by job_type pattern (for stuck training, etc.)
            if job_type and killed == 0:
                patterns = {
                    "training": ["train_nnue", "train.*model"],
                    "selfplay": ["selfplay", "run_hybrid_selfplay"],
                }

                def _pkill_pattern(pattern: str) -> bool:
                    """Run pkill in thread pool to avoid blocking event loop."""
                    result = subprocess.run(
                        ["pkill", "-9", "-f", pattern],
                        timeout=5,
                        capture_output=True,
                    )
                    return result.returncode == 0

                for pattern in patterns.get(job_type, [job_type]):
                    try:
                        # Jan 24, 2026: Run in thread to avoid blocking event loop
                        import asyncio
                        success = await asyncio.to_thread(_pkill_pattern, pattern)
                        if success:
                            killed += 1
                            logger.info(f"Killed processes matching '{pattern}': {reason}")
                    except Exception as e:  # noqa: BLE001
                        logger.info(f"pkill error for {pattern}: {e}")

            return web.json_response({
                "success": killed > 0,
                "killed": killed,
                "reason": reason,
            })

        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=400)

    async def handle_cleanup(self, request: web.Request) -> web.Response:
        """Handle cleanup request (from leader or manual).

        LEARNED LESSONS - This endpoint allows remote nodes to trigger disk cleanup
        when the leader detects disk usage approaching critical thresholds.

        Returns:
            success: True if cleanup was initiated
            disk_percent_before: Disk usage before cleanup
            message: Status message
        """
        try:
            logger.info("Cleanup request received")

            # Run cleanup in background to avoid blocking the request
            safe_create_task(self._cleanup_local_disk(), name="jobs-cleanup-disk")

            # Return current disk usage
            usage = self._get_resource_usage()
            return web.json_response({
                "success": True,
                "disk_percent_before": usage["disk_percent"],
                "message": "Cleanup initiated",
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    async def handle_restart_stuck_jobs(self, request: web.Request) -> web.Response:
        """Handle request to restart stuck selfplay jobs.

        LEARNED LESSONS - Called by leader when it detects GPU idle with running processes.
        Kills all selfplay processes and clears job tracking so they restart.

        Returns:
            success: True if restart was initiated
            message: Status message
        """
        try:
            logger.info("Restart stuck jobs request received")

            # Run in background to avoid blocking
            safe_create_task(self._restart_local_stuck_jobs(), name="jobs-restart-stuck")

            return web.json_response({
                "success": True,
                "message": "Stuck job restart initiated",
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    async def handle_cleanup_files(self, request: web.Request) -> web.Response:
        """Delete specific files from this node (for post-sync cleanup).

        Called by leader after successful sync to training nodes to free
        disk space on source nodes with high disk usage.

        Request body:
            files: List of file paths to delete (relative to data directory)
            reason: Reason for cleanup

        Returns:
            success: True if any files were deleted
            freed_bytes: Total bytes freed
            deleted_count: Number of files deleted
        """
        try:
            data = await request.json()
            files = data.get("files", [])
            reason = data.get("reason", "manual")

            if not files:
                return web.json_response(
                    {"success": False, "error": "No files specified"},
                    status=400,
                )

            logger.info(f"Cleanup files request: {len(files)} files, reason={reason}")

            data_dir = self.get_data_directory()
            freed_bytes = 0
            deleted_count = 0

            for file_path in files:
                # Security: only allow deletion within data directory
                full_path = data_dir / (file_path or "").lstrip("/")
                try:
                    data_root = data_dir.resolve()
                    resolved = full_path.resolve()
                    resolved.relative_to(data_root)
                except (AttributeError, ValueError):
                    logger.info(f"Cleanup: skipping path outside data dir: {file_path}")
                    continue

                if resolved.exists():
                    try:
                        size = resolved.stat().st_size
                        resolved.unlink()
                        freed_bytes += size
                        deleted_count += 1
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"Failed to delete {file_path}: {e}")

            logger.info(f"Cleanup complete: {deleted_count} files, {freed_bytes / 1e6:.1f}MB freed")

            return web.json_response({
                "success": True,
                "freed_bytes": freed_bytes,
                "deleted_count": deleted_count,
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)
