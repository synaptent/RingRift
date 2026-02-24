"""Improvement Loop HTTP Handlers Mixin.

Provides HTTP endpoints for AlphaZero-style improvement loops.
Handles loop start, status, phase completion, and evaluation coordination.

Usage:
    class P2POrchestrator(ImprovementHandlersMixin, ...):
        pass

Endpoints:
    POST /improvement/start - Start an improvement loop
    GET /improvement/status - Get status of improvement loops
    POST /improvement/phase_complete - Notify phase completion
    GET /improvement_cycles/status - Get status via ImprovementCycleManager
    GET /improvement_cycles/leaderboard - Get Elo leaderboard
    POST /improvement_cycles/training_complete - Report training completion
    POST /improvement_cycles/evaluation_complete - Report evaluation completion

Requires the implementing class to have:
    - role: NodeRole
    - leader_id: str | None
    - node_id: str
    - peers: Dict[str, PeerInfo]
    - peers_lock: threading.Lock
    - improvement_loop_state: Dict[str, ImprovementLoopState]
    - improvement_cycle_manager: ImprovementCycleManager | None
    - _run_improvement_loop(job_id) method
    - _schedule_improvement_evaluation(cycle_id, model_id) method
    - _auto_deploy_model(model_path, board_type, num_players) method
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from aiohttp import web

from app.core.async_context import safe_create_task
from scripts.p2p.handlers.base import BaseP2PHandler
from scripts.p2p.handlers.timeout_decorator import handler_timeout, HANDLER_TIMEOUT_TOURNAMENT

if TYPE_CHECKING:
    from scripts.p2p.managers.state_manager import ImprovementLoopState

logger = logging.getLogger(__name__)

# Try to import rate negotiation - optional dependency
try:
    from app.coordination.resource_optimizer import negotiate_selfplay_rate
    HAS_RATE_NEGOTIATION = True
except ImportError:
    HAS_RATE_NEGOTIATION = False
    negotiate_selfplay_rate = None


class ImprovementHandlersMixin(BaseP2PHandler):
    """Mixin providing improvement loop HTTP handlers.

    Inherits from BaseP2PHandler for consistent response formatting.

    Requires the implementing class to have:
    - role: NodeRole (LEADER, FOLLOWER, etc.)
    - leader_id: str | None
    - node_id: str
    - peers: Dict[str, PeerInfo]
    - peers_lock: threading.Lock
    - improvement_loop_state: Dict[str, ImprovementLoopState]
    - improvement_cycle_manager: ImprovementCycleManager | None
    - _run_improvement_loop(job_id) method
    - _schedule_improvement_evaluation(cycle_id, model_id) method
    - _auto_deploy_model(model_path, board_type, num_players) method
    """

    # Type hints for IDE support
    role: Any  # NodeRole
    leader_id: str | None
    peers: dict
    peers_lock: object  # threading.Lock
    improvement_loop_state: dict
    improvement_cycle_manager: Any

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_improvement_start(self, request: web.Request) -> web.Response:
        """POST /improvement/start - Start an improvement loop (AlphaZero-style training cycle).

        Only the leader can start improvement loops.

        Request body:
            board_type: Board type (default: "square8")
            num_players: Number of players (default: 2)
            max_iterations: Maximum iterations (default: 50)
            games_per_iteration: Games per iteration (default: 1000)

        Returns:
            job_id: Improvement loop job ID
            workers: List of available worker nodes
            gpu_workers: List of GPU workers for training
            config: Loop configuration
        """
        try:
            # Import NodeRole lazily to avoid circular imports
            from scripts.p2p_orchestrator import NodeRole

            if self.role != NodeRole.LEADER:
                return web.json_response({
                    "error": "Only the leader can start improvement loops",
                    "leader_id": self.leader_id,
                }, status=403)

            data = await request.json()
            job_id = f"improve_{uuid.uuid4().hex[:8]}"

            # Query negotiated rate from resource_optimizer for cooperative utilization
            # This ensures selfplay rate respects cluster-wide 60-80% utilization target
            requested_games = data.get("games_per_iteration", 1000)
            if HAS_RATE_NEGOTIATION and negotiate_selfplay_rate is not None:
                try:
                    # Negotiate rate with resource_optimizer (60-80% target)
                    approved_rate = negotiate_selfplay_rate(
                        requested_rate=requested_games,
                        reason=f"p2p_improvement_loop:{job_id}",
                        requestor=f"p2p_{self.node_id}",
                    )
                    if approved_rate != requested_games:
                        logger.info(f"games_per_iteration adjusted: {requested_games} -> {approved_rate} (utilization-based)")
                    requested_games = approved_rate
                except Exception as e:  # noqa: BLE001
                    logger.info(f"Rate negotiation failed, using default: {e}")

            # Import ImprovementLoopState lazily
            from scripts.p2p.managers.state_manager import ImprovementLoopState

            state = ImprovementLoopState(
                job_id=job_id,
                board_type=data.get("board_type", "square8"),
                num_players=data.get("num_players", 2),
                max_iterations=data.get("max_iterations", 50),
                games_per_iteration=requested_games,
                phase="selfplay",
                status="running",
                started_at=time.time(),
                last_update=time.time(),
            )

            # Find available workers
            with self.peers_lock:
                workers = [p.node_id for p in self.peers.values() if p.is_healthy()]
                gpu_workers = [p.node_id for p in self.peers.values() if p.is_healthy() and p.has_gpu]
            state.worker_nodes = workers

            if not gpu_workers:
                return web.json_response({"error": "No GPU workers available for training"}, status=503)

            self.improvement_loop_state[job_id] = state

            logger.info(f"Started improvement loop {job_id}: {len(workers)} workers, {len(gpu_workers)} GPU workers")

            # Launch improvement loop
            safe_create_task(self._run_improvement_loop(job_id), name="improvement-loop-run")

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "workers": workers,
                "gpu_workers": gpu_workers,
                "config": {
                    "board_type": state.board_type,
                    "num_players": state.num_players,
                    "max_iterations": state.max_iterations,
                    "games_per_iteration": state.games_per_iteration,
                },
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_improvement_status(self, request: web.Request) -> web.Response:
        """GET /improvement/status - Get status of improvement loops.

        Query params:
            job_id: Specific job ID (optional). If not provided, returns all loops.

        Returns:
            Single loop status or dict of all loop statuses.
        """
        try:
            job_id = request.query.get("job_id")

            if job_id:
                if job_id not in self.improvement_loop_state:
                    return web.json_response({"error": "Improvement loop not found"}, status=404)
                state = self.improvement_loop_state[job_id]
                return web.json_response(state.to_dict())

            return web.json_response({
                job_id: state.to_dict()
                for job_id, state in self.improvement_loop_state.items()
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_improvement_phase_complete(self, request: web.Request) -> web.Response:
        """POST /improvement/phase_complete - Notify that a phase of the improvement loop is complete.

        Request body:
            job_id: Improvement loop job ID
            phase: Phase that completed ("selfplay", "train", "evaluate")
            worker_id: ID of the worker that completed the phase
            result: Phase-specific result data

        Returns:
            success: Whether the notification was recorded
            job_id: The improvement loop job ID
            phase: Current phase after update
            iteration: Current iteration number
        """
        try:
            data = await request.json()
            job_id = data.get("job_id")
            phase = data.get("phase")
            worker_id = data.get("worker_id", "unknown")
            result = data.get("result", {})

            if job_id not in self.improvement_loop_state:
                return web.json_response({"error": "Improvement loop not found"}, status=404)

            state = self.improvement_loop_state[job_id]
            state.last_update = time.time()

            # Track progress by phase
            if phase == "selfplay":
                games_done = result.get("games_done", 0)
                state.selfplay_progress[worker_id] = games_done
                total_done = sum(state.selfplay_progress.values())
                logger.info(f"Improvement loop selfplay: {total_done}/{state.games_per_iteration} games")
            elif phase == "train":
                state.best_model_path = result.get("model_path", state.best_model_path)
            elif phase == "evaluate":
                winrate = result.get("winrate", 0.0)
                if winrate > state.best_winrate:
                    state.best_winrate = winrate
                    logger.info(f"New best model: winrate={winrate:.2%}")

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "phase": state.phase,
                "iteration": state.current_iteration,
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_improvement_cycles_status(self, request: web.Request) -> web.Response:
        """GET /improvement_cycles/status - Get status of all improvement cycles.

        Uses ImprovementCycleManager for cycle-based improvements.

        Returns:
            success: True if manager is available
            is_leader: Whether this node is the leader
            ...status fields from ImprovementCycleManager.get_status()
        """
        try:
            if not self.improvement_cycle_manager:
                return web.json_response({
                    "success": False,
                    "error": "ImprovementCycleManager not initialized"
                })

            status = self.improvement_cycle_manager.get_status()

            # Import NodeRole lazily
            from scripts.p2p_orchestrator import NodeRole

            return web.json_response({
                "success": True,
                "is_leader": self.role == NodeRole.LEADER,
                **status,
            })

        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)})

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_improvement_cycles_leaderboard(self, request: web.Request) -> web.Response:
        """GET /improvement_cycles/leaderboard - Get Elo leaderboard.

        Query params:
            board_type: Filter by board type (optional)
            num_players: Filter by number of players (optional)

        Returns:
            success: True if manager is available
            leaderboard: List of model entries with Elo ratings
            total_models: Total number of models in leaderboard
        """
        try:
            if not self.improvement_cycle_manager:
                return web.json_response({
                    "success": False,
                    "error": "ImprovementCycleManager not initialized"
                })

            board_type = request.query.get("board_type")
            num_players_str = request.query.get("num_players")
            num_players = int(num_players_str) if num_players_str else None

            leaderboard = self.improvement_cycle_manager.get_leaderboard(
                board_type=board_type,
                num_players=num_players,
            )

            return web.json_response({
                "success": True,
                "leaderboard": [e.to_dict() for e in leaderboard],
                "total_models": len(leaderboard),
            })

        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)})

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_improvement_training_complete(self, request: web.Request) -> web.Response:
        """POST /improvement_cycles/training_complete - Report training completion.

        Request body:
            cycle_id: Improvement cycle ID
            model_id: ID of the trained model
            model_path: Path to the trained model (optional)
            success: Whether training succeeded
            error: Error message if training failed (optional)

        If training succeeded and this node is leader, schedules evaluation.

        Returns:
            success: True if notification was recorded
        """
        try:
            if not self.improvement_cycle_manager:
                return web.json_response({"success": False, "error": "ImprovementCycleManager not initialized"})

            data = await request.json()
            cycle_id = data.get("cycle_id")
            new_model_id = data.get("model_id")
            model_path = data.get("model_path", "")
            success = data.get("success", False)
            error_message = data.get("error", "")

            self.improvement_cycle_manager.handle_training_complete(
                cycle_id=cycle_id, new_model_id=new_model_id, model_path=model_path,
                success=success, error_message=error_message,
            )

            # Import NodeRole lazily
            from scripts.p2p_orchestrator import NodeRole

            if success and self.role == NodeRole.LEADER:
                safe_create_task(self._schedule_improvement_evaluation(cycle_id, new_model_id), name="improvement-schedule-eval")

            return web.json_response({"success": True})

        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)})

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_improvement_evaluation_complete(self, request: web.Request) -> web.Response:
        """POST /improvement_cycles/evaluation_complete - Report evaluation completion.

        Request body:
            cycle_id: Improvement cycle ID
            model_id: ID of the evaluated model
            best_model_id: ID of the current best model
            wins: Number of wins
            losses: Number of losses
            draws: Number of draws
            model_path: Path to the model (optional)
            board_type: Board type (optional, for auto-deploy)
            num_players: Number of players (optional, for auto-deploy)

        If the new model is best, triggers auto-deployment.

        Returns:
            success: True if notification was recorded
        """
        try:
            if not self.improvement_cycle_manager:
                return web.json_response({"success": False, "error": "ImprovementCycleManager not initialized"})

            data = await request.json()
            self.improvement_cycle_manager.handle_evaluation_complete(
                cycle_id=data.get("cycle_id"), new_model_id=data.get("model_id"),
                best_model_id=data.get("best_model_id"), wins=data.get("wins", 0),
                losses=data.get("losses", 0), draws=data.get("draws", 0),
            )

            # Auto-deploy model if evaluation passed (new model is best)
            if data.get("model_id") == data.get("best_model_id"):
                model_path = data.get("model_path", "")
                board_type = data.get("board_type", "square8")
                num_players = data.get("num_players", 2)
                if model_path:
                    safe_create_task(self._auto_deploy_model(model_path, board_type, num_players), name="improvement-auto-deploy")

            return web.json_response({"success": True})

        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)})
