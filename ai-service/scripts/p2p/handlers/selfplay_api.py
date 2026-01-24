"""Selfplay API HTTP handlers for P2P orchestrator.

January 2026 - P2P Modularization Phase 1b

This mixin provides HTTP handlers for selfplay-related API endpoints.

Must be mixed into a class that provides:
- self.is_partition_readonly() -> bool
- self.get_partition_status() -> dict
- self.job_manager: JobManager with run_gpu_selfplay_job()
- self.selfplay_scheduler: SelfplayScheduler with track_diversity()
- self.node_id: str
- self.is_leader: bool
- self._proxy_to_leader(request) -> web.Response
- self._reduce_local_selfplay_jobs(target, reason) -> dict
- self._run_local_canonical_selfplay(job_id, board_type, num_players, num_games, seed)
- self._is_leader() -> bool
- self.manifest_lock: threading.RLock
- self.cluster_data_manifest: Optional[ClusterDataManifest]
- self.local_data_manifest: Optional[LocalDataManifest]
- self.selfplay_stats_history: list
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any

from aiohttp import web

# Jan 2026: Allow selfplay in degraded mode (quorum loss doesn't block local GPU work)
# Selfplay is a local operation that doesn't require cluster consensus.
# Set RINGRIFT_ALLOW_DEGRADED_SELFPLAY=true to enable (default: true)
ALLOW_DEGRADED_SELFPLAY = os.environ.get(
    "RINGRIFT_ALLOW_DEGRADED_SELFPLAY", "true"
).lower() in ("true", "1", "yes")

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SelfplayHandlersMixin:
    """Mixin providing selfplay API HTTP handlers.

    Endpoints:
    - POST /reduce_selfplay - Load shedding, reduce selfplay jobs
    - POST /selfplay/start - Start GPU selfplay job on this node
    - POST /dispatch_selfplay - Request selfplay jobs for a config
    - POST /pipeline/selfplay_worker - Worker endpoint for canonical selfplay
    - GET /api/selfplay_stats - Get aggregated selfplay statistics
    """

    async def handle_reduce_selfplay(self, request: web.Request) -> web.Response:
        """Stop excess selfplay jobs on this node (load shedding).

        Used by leaders when a node is under memory/disk pressure so the node
        can recover without requiring manual intervention.
        """
        try:
            data = await request.json()
            target_raw = data.get("target_selfplay_jobs", data.get("target", 0))
            reason = str(data.get("reason") or "remote_request")
            try:
                target = int(target_raw)
            except (ValueError):
                target = 0

            result = await self._reduce_local_selfplay_jobs(target, reason=reason)
            return web.json_response({"success": True, **result})
        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)}, status=400)

    def _can_dispatch_selfplay_degraded(self) -> tuple[bool, str]:
        """Check if selfplay can be dispatched in degraded cluster state.

        January 2026: Selfplay is a local GPU operation that doesn't require
        cluster consensus. Allow it even when quorum is lost, as long as the
        local node is healthy.

        Returns:
            (can_dispatch, reason) - True if selfplay can proceed locally
        """
        # Check local resource health
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 95:
                return False, f"Local memory usage too high: {memory_percent:.1f}%"

            # Check GPU memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        mem_info = torch.cuda.memory_stats(i)
                        if "allocated_bytes.all.current" in mem_info:
                            allocated = mem_info["allocated_bytes.all.current"]
                            total = torch.cuda.get_device_properties(i).total_memory
                            if allocated / total > 0.95:
                                return False, f"GPU {i} memory too high: {allocated/total:.1%}"
            except (ImportError, RuntimeError):
                pass  # No GPU or torch not available

        except ImportError:
            pass  # psutil not available

        return True, "Local health OK for degraded-mode selfplay"

    def _get_gpu_count_sync(self) -> int:
        """Get the number of CUDA GPUs available (blocking version).

        Jan 18, 2026: Added for multi-GPU selfplay support.
        Jan 24, 2026: Renamed to _sync, use _get_gpu_count_async() from async context.

        Returns:
            Number of CUDA GPUs, or 1 if detection fails or no CUDA available.
        """
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except ImportError:
            pass  # PyTorch not available

        # Fallback: try nvidia-smi
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                gpu_names = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
                return len(gpu_names) if gpu_names else 1
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

        return 1  # Default to single GPU

    async def _get_gpu_count_async(self) -> int:
        """Get GPU count without blocking the event loop.

        Jan 24, 2026: Async wrapper to prevent event loop blocking from
        subprocess.run() calls in nvidia-smi fallback path.
        """
        import asyncio
        return await asyncio.to_thread(self._get_gpu_count_sync)

    async def handle_selfplay_start(self, request: web.Request) -> web.Response:
        """POST /selfplay/start - Start GPU selfplay job on this node.

        Called by leader to dispatch GPU selfplay work to worker nodes.
        Uses run_hybrid_selfplay.py for GPU-accelerated game generation.

        January 2026: Now supports degraded mode operation when ALLOW_DEGRADED_SELFPLAY
        is enabled. Selfplay doesn't require cluster consensus - it's local GPU work.
        """
        try:
            # Phase 2.4 (Dec 29, 2025): Block dispatch if in partition readonly mode
            # Jan 2026: Allow bypass for selfplay (doesn't require consensus)
            if self.is_partition_readonly():
                status = self.get_partition_status()

                # Check if degraded selfplay is allowed
                if ALLOW_DEGRADED_SELFPLAY:
                    can_proceed, reason = self._can_dispatch_selfplay_degraded()
                    if can_proceed:
                        logger.info(
                            f"[P2P] Allowing selfplay in degraded mode: "
                            f"partition_status={status['partition_status']}, reason={reason}"
                        )
                        # Continue with selfplay - don't return 503
                    else:
                        logger.warning(
                            f"[P2P] Rejecting selfplay in degraded mode: {reason}"
                        )
                        return web.json_response({
                            "success": False,
                            "error": f"Degraded mode selfplay blocked: {reason}",
                            "partition_status": status["partition_status"],
                            "health_ratio": status["health_ratio"],
                            "retry_after_seconds": 30,
                        }, status=503)
                else:
                    logger.warning(
                        f"[P2P] Rejecting selfplay start: partition readonly mode "
                        f"(status={status['partition_status']}, ratio={status['health_ratio']:.2%})"
                    )
                    return web.json_response({
                        "success": False,
                        "error": "Node is in partition readonly mode",
                        "partition_status": status["partition_status"],
                        "health_ratio": status["health_ratio"],
                        "retry_after_seconds": 60,
                    }, status=503)

            data = await request.json()
            board_type = data.get("board_type", "square8")
            num_players = data.get("num_players", 2)
            num_games = data.get("num_games", 500)
            engine_mode = data.get("engine_mode", "gumbel-mcts")  # GPU-accelerated MCTS
            engine_extra_args = data.get("engine_extra_args")  # December 2025: for budget override
            # Jan 21, 2026: Default to v2 (most canonical models are v2)
            model_version = data.get("model_version", "v2")
            data.get("auto_scaled", False)

            job_id = f"selfplay-{self.node_id}-{int(time.time())}"

            # Jan 18, 2026: Multi-GPU support - spawn one worker per GPU
            # This improves utilization on multi-GPU Vast.ai nodes (8x3090, 4x5090, etc.)
            num_gpus = await self._get_gpu_count_async()
            if num_gpus > 1:
                # Distribute games across GPUs
                games_per_gpu = max(1, num_games // num_gpus)
                remainder = num_games % num_gpus

                logger.info(
                    f"Multi-GPU node detected ({num_gpus} GPUs), "
                    f"spawning {num_gpus} parallel workers"
                )

                for gpu_id in range(num_gpus):
                    gpu_games = games_per_gpu + (1 if gpu_id < remainder else 0)
                    gpu_job_id = f"{job_id}_gpu{gpu_id}"

                    # Start worker for this GPU
                    asyncio.create_task(self.job_manager.run_gpu_selfplay_job(
                        job_id=gpu_job_id,
                        board_type=board_type,
                        num_players=num_players,
                        num_games=gpu_games,
                        engine_mode=engine_mode,
                        engine_extra_args=engine_extra_args,
                        model_version=model_version,
                        cuda_device=gpu_id,  # Route to specific GPU
                    ))

                logger.info(
                    f"Started {num_gpus} GPU selfplay workers {job_id}_gpu*: "
                    f"{board_type}/{num_players}p, {num_games} total games"
                )
            else:
                # Single GPU or CPU-only: start single job (original behavior)
                asyncio.create_task(self.job_manager.run_gpu_selfplay_job(
                    job_id=job_id,
                    board_type=board_type,
                    num_players=num_players,
                    num_games=num_games,
                    engine_mode=engine_mode,
                    engine_extra_args=engine_extra_args,
                    model_version=model_version,
                ))

                logger.info(f"Started GPU selfplay job {job_id}: {board_type}/{num_players}p, {num_games} games")

            # Track diversity metrics for monitoring (Phase 2B, Dec 2025)
            # Jan 2026: Resolve mode-specific engines (minimax-only -> brs/maxn/minimax)
            # This ensures BRS and MaxN appear as distinct tracked modes
            resolved_engine, _ = self.selfplay_scheduler.resolve_engine_mode(
                engine_mode, has_gpu=True
            )
            self.selfplay_scheduler.track_diversity({
                "board_type": board_type,
                "num_players": num_players,
                "engine_mode": resolved_engine,
            })

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "board_type": board_type,
                "num_players": num_players,
                "num_games": num_games,
                "node_id": self.node_id,
            })
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to start selfplay: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_dispatch_selfplay(self, request: web.Request) -> web.Response:
        """POST /dispatch_selfplay - Request selfplay jobs for a config.

        Called from coordinator or any node to request selfplay generation.
        If this node is leader, adds work to the queue directly.
        If not leader, proxies the request to the current leader.

        Request body:
        {
            "board_type": "hex8",
            "num_players": 4,
            "num_games": 200,
            "engine_mode": "gumbel-mcts",  # optional, defaults to gumbel-mcts
            "priority": 50,  # optional, 1-100, higher = more urgent
            "force": false  # optional, bypass backpressure
        }

        Response:
        {
            "success": true,
            "work_ids": ["abc123", ...],
            "count": 2,
            "dispatched_to": "leader-node-id"
        }
        """
        try:
            data = await request.json()
        except (json.JSONDecodeError, ValueError):
            return web.json_response(
                {"success": False, "error": "Invalid JSON body"},
                status=400,
            )

        board_type = data.get("board_type", "square8")
        num_players = data.get("num_players", 2)
        num_games = data.get("num_games", 200)
        engine_mode = data.get("engine_mode", "gumbel-mcts")
        # Jan 4, 2026: Convert priority from string or int to int
        priority_raw = data.get("priority", 50)
        if isinstance(priority_raw, str):
            priority_map = {"low": 25, "normal": 50, "high": 75, "critical": 100}
            priority = priority_map.get(priority_raw.lower(), 50)
        else:
            priority = int(priority_raw)
        force = data.get("force", False)

        # If we're the leader, add work directly to queue
        if self.is_leader:
            try:
                from app.coordination.work_queue import get_work_queue, WorkItem, WorkType

                wq = get_work_queue()
                if wq is None:
                    return web.json_response(
                        {"success": False, "error": "Work queue not available"},
                        status=503,
                    )

                # Calculate number of work items (split into 100-game chunks for better distribution)
                games_per_job = 100
                num_jobs = max(1, num_games // games_per_job)
                games_remainder = num_games % games_per_job

                work_ids = []
                for i in range(num_jobs):
                    games_this_job = games_per_job if i < num_jobs - 1 or games_remainder == 0 else games_remainder
                    if games_this_job == 0 and i == num_jobs - 1:
                        games_this_job = games_per_job  # Use full batch for last if no remainder

                    config_key = f"{board_type}_{num_players}p"
                    item = WorkItem(
                        work_type=WorkType.SELFPLAY,
                        priority=priority,
                        config={
                            "board_type": board_type,
                            "num_players": num_players,
                            "num_games": games_this_job,
                            "engine_mode": engine_mode,
                            "config_key": config_key,
                            # Dec 30, 2025: Prevent coordinator/CPU nodes from claiming selfplay
                            "requires_gpu": True,
                        },
                        timeout_seconds=3600.0,
                    )
                    work_id = wq.add_work(item, force=force)
                    work_ids.append(work_id)

                logger.info(
                    f"[P2P] Dispatched {len(work_ids)} selfplay jobs for {board_type}_{num_players}p "
                    f"({num_games} total games, engine={engine_mode})"
                )

                return web.json_response({
                    "success": True,
                    "work_ids": work_ids,
                    "count": len(work_ids),
                    "total_games": num_games,
                    "dispatched_to": self.node_id,
                    "config_key": f"{board_type}_{num_players}p",
                })

            except RuntimeError as e:
                if "BACKPRESSURE" in str(e):
                    logger.warning(f"Selfplay dispatch rejected due to backpressure: {e}")
                    return web.json_response(
                        {"success": False, "error": str(e), "retry_after_seconds": 60},
                        status=429,
                    )
                raise
            except Exception as e:
                logger.error(f"Failed to dispatch selfplay: {e}")
                return web.json_response({"success": False, "error": str(e)}, status=500)

        # Not leader - proxy to leader
        return await self._proxy_to_leader(request)

    async def handle_pipeline_selfplay_worker(self, request: web.Request) -> web.Response:
        """POST /pipeline/selfplay_worker - Worker endpoint for canonical selfplay."""
        try:
            data = await request.json()
            asyncio.create_task(self._run_local_canonical_selfplay(
                data.get("job_id"), data.get("board_type", "square8"), data.get("num_players", 2),
                data.get("num_games", 500), data.get("seed", 0)))
            return web.json_response({"success": True, "job_id": data.get("job_id"),
                                     "message": f"Started canonical selfplay: {data.get('num_games', 500)} games"})
        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_api_selfplay_stats(self, request: web.Request) -> web.Response:
        """Get aggregated selfplay game statistics for dashboard charts."""
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                proxied = await self._proxy_to_leader(request)
                if proxied.status not in (502, 503):
                    return proxied

            with self.manifest_lock:
                cluster_manifest = self.cluster_data_manifest
                local_manifest = self.local_data_manifest
                history = list(self.selfplay_stats_history)

            by_board_type: dict[str, dict[str, Any]] = {}
            total_selfplay_games = 0
            manifest_collected_at = 0.0

            if cluster_manifest:
                by_board_type = cluster_manifest.by_board_type
                total_selfplay_games = int(cluster_manifest.total_selfplay_games or 0)
                manifest_collected_at = float(cluster_manifest.collected_at or 0.0)
            elif local_manifest:
                manifest_collected_at = float(local_manifest.collected_at or 0.0)
                totals: dict[str, int] = {}
                for f in getattr(local_manifest, "files", []) or []:
                    if getattr(f, "file_type", "") != "selfplay":
                        continue
                    board_type = getattr(f, "board_type", "") or ""
                    num_players = int(getattr(f, "num_players", 0) or 0)
                    if not board_type or not num_players:
                        continue
                    key = f"{board_type}_{num_players}p"
                    totals[key] = totals.get(key, 0) + int(getattr(f, "game_count", 0) or 0)
                by_board_type = {k: {"total_games": v, "nodes": [local_manifest.node_id]} for k, v in totals.items()}
                total_selfplay_games = sum(totals.values())

            return web.json_response(
                {
                    "success": True,
                    "node_id": self.node_id,
                    "is_leader": self._is_leader(),
                    "manifest_collected_at": manifest_collected_at,
                    "total_selfplay_games": total_selfplay_games,
                    "by_board_type": by_board_type,
                    "history": history,
                    "timestamp": time.time(),
                }
            )
        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)}, status=500)
