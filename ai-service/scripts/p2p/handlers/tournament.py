"""Tournament HTTP Handlers Mixin.

Provides HTTP endpoints for distributed round-robin tournaments.
Enables comprehensive model evaluation by playing all models against
each other across cluster nodes.

Usage:
    class P2POrchestrator(TournamentHandlersMixin, ...):
        pass

Endpoints:
    POST /tournament/start - Start distributed tournament (leader only)
    GET /tournament/status - Get tournament progress and standings
    POST /tournament/match - Execute a single tournament match (worker)
    GET /tournament/results - Get final tournament results and rankings
    POST /tournament/stop - Cancel running tournament

Tournament Format:
    Round-robin: Each model plays against every other model.
    Matches distributed to workers for parallel execution.
    Results aggregated to compute Elo ratings and final standings.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from aiohttp import web

from scripts.p2p.handlers.base import BaseP2PHandler
from scripts.p2p.handlers.timeout_decorator import (
    handler_timeout,
    HANDLER_TIMEOUT_TOURNAMENT,
    HANDLER_TIMEOUT_GOSSIP,
)

if TYPE_CHECKING:
    from scripts.p2p.models import NodeRole

logger = logging.getLogger(__name__)


class TournamentHandlersMixin(BaseP2PHandler):
    """Mixin providing distributed tournament HTTP handlers.

    Inherits from BaseP2PHandler for consistent response formatting.

    Requires the implementing class to have:
    - node_id: str
    - role: NodeRole
    - peers: dict
    - peers_lock: threading.Lock
    - distributed_tournament_state: dict
    - job_manager: JobManager (with run_distributed_tournament method)
    - _play_tournament_match() method
    """

    # Type hints for IDE support
    node_id: str
    role: Any  # NodeRole
    peers: dict
    peers_lock: Any
    distributed_tournament_state: dict
    job_manager: Any  # JobManager

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_tournament_start(self, request: web.Request) -> web.Response:
        """Start or propose a distributed tournament.

        DISTRIBUTED TOURNAMENT SCHEDULING:
        - Leaders can start tournaments directly (immediate)
        - Non-leaders can propose tournaments (gossip-based consensus)

        Request body:
        {
            "board_type": "square8",
            "num_players": 2,
            "agent_ids": ["agent1", "agent2", "agent3"],
            "games_per_pairing": 2
        }
        """
        from scripts.p2p.models import DistributedTournamentState
        from scripts.p2p.types import NodeRole

        try:
            data = await request.json()

            # Non-leaders forward tournament requests to the leader
            if self.role != NodeRole.LEADER:
                # Find the leader's endpoint and forward the request
                leader_id = getattr(self, 'leader_id', None)
                if leader_id and leader_id in self.peers:
                    leader_info = self.peers[leader_id]
                    leader_url = getattr(leader_info, 'url', None)
                    if leader_url:
                        import aiohttp
                        try:
                            async with aiohttp.ClientSession() as session:
                                async with session.post(
                                    f"{leader_url.rstrip('/')}/tournament/start",
                                    json=data,
                                    timeout=aiohttp.ClientTimeout(total=30)
                                ) as resp:
                                    result = await resp.json()
                                    return self.json_response(result)
                        except Exception as e:
                            logger.warning(f"Failed to forward to leader: {e}")

                return self.error_response(
                    "This node is not the leader. Please send tournament requests to the leader.",
                    status=400
                )

            # Leader can start tournaments directly
            job_id = f"tournament_{uuid.uuid4().hex[:8]}"

            agent_ids = data.get("agent_ids", [])
            if len(agent_ids) < 2:
                return self.error_response("At least 2 agents required", status=400)

            # Create round-robin pairings
            pairings = []
            for i, a1 in enumerate(agent_ids):
                for a2 in agent_ids[i+1:]:
                    for game_num in range(data.get("games_per_pairing", 2)):
                        pairings.append({
                            "agent1": a1,
                            "agent2": a2,
                            "game_num": game_num,
                            "status": "pending",
                        })

            state = DistributedTournamentState(
                job_id=job_id,
                board_type=data.get("board_type", "square8"),
                num_players=data.get("num_players", 2),
                agent_ids=agent_ids,
                games_per_pairing=data.get("games_per_pairing", 2),
                total_matches=len(pairings),
                pending_matches=pairings,
                status="running",
                started_at=time.time(),
                last_update=time.time(),
            )

            # Find available workers
            with self.peers_lock:
                workers = [p.node_id for p in self.peers.values() if p.is_healthy()]
            state.worker_nodes = workers

            if not state.worker_nodes:
                return self.error_response("No workers available", status=503)

            self.distributed_tournament_state[job_id] = state

            logger.info(f"Started tournament {job_id}: {len(agent_ids)} agents, {len(pairings)} matches, {len(workers)} workers")

            # Launch coordinator task via JobManager
            # Dec 28, 2025: Added error callback to log task failures
            async def _run_tournament_with_logging():
                try:
                    await self.job_manager.run_distributed_tournament(job_id)
                except Exception as e:
                    logger.exception(f"Tournament coordinator task failed for {job_id}: {e}")

            asyncio.create_task(_run_tournament_with_logging())

            return self.json_response({
                "success": True,
                "job_id": job_id,
                "agents": agent_ids,
                "total_matches": len(pairings),
                "workers": workers,
            })
        except Exception as e:
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_tournament_match(self, request: web.Request) -> web.Response:
        """Execute a tournament match and return results synchronously.

        Dec 28, 2025: Fixed to run match synchronously and return results in
        response. The coordinator waits for results, so fire-and-forget was
        causing 0 completed matches.
        """
        try:
            data = await request.json()
            job_id = data.get("job_id")
            match_info = data.get("match")

            if not job_id or not match_info:
                return self.error_response("job_id and match required", status=400)

            logger.info(f"Received tournament match request: {match_info}")

            # Run match synchronously and return result
            # The coordinator expects results in the response
            result = await self._play_tournament_match(job_id, match_info)

            return self.json_response({
                "success": True,
                "job_id": job_id,
                "status": "match_completed",
                "results": [result] if result else [],
            })
        except Exception as e:
            logger.error(f"Tournament match error: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_tournament_status(self, request: web.Request) -> web.Response:
        """Get status of distributed tournaments."""
        try:
            job_id = request.query.get("job_id")

            if job_id:
                if job_id not in self.distributed_tournament_state:
                    return self.error_response("Tournament not found", status=404)
                state = self.distributed_tournament_state[job_id]
                return self.json_response(state.to_dict())

            return self.json_response({
                job_id: state.to_dict()
                for job_id, state in self.distributed_tournament_state.items()
            })
        except Exception as e:
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_tournament_result(self, request: web.Request) -> web.Response:
        """Receive match result from a worker."""
        try:
            data = await request.json()
            job_id = data.get("job_id")
            match_result = data.get("result", {})
            worker_id = data.get("worker_id", "unknown")

            if job_id not in self.distributed_tournament_state:
                return self.error_response("Tournament not found", status=404)

            state = self.distributed_tournament_state[job_id]
            state.results.append(match_result)
            state.completed_matches += 1
            state.last_update = time.time()

            logger.info(f"Tournament result: {state.completed_matches}/{state.total_matches} matches from {worker_id}")

            return self.json_response({
                "success": True,
                "job_id": job_id,
                "completed": state.completed_matches,
                "total": state.total_matches,
            })
        except Exception as e:
            return self.error_response(str(e), status=500)
