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

    # =========================================================================
    # Generation Tournament Handlers (January 2026)
    # =========================================================================

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_generation_tournament(self, request: web.Request) -> web.Response:
        """Execute a generation head-to-head tournament on this node.

        Runs a tournament between parent and child generations to validate
        training improvement. Uses Wilson CI for statistical significance.

        Request body:
        {
            "parent_generation_id": 1,
            "child_generation_id": 2,
            "num_games": 100,
            "harness_type": "gumbel_mcts",
            "difficulty": 1.0
        }

        Response:
        {
            "status": "completed",
            "stats": {
                "parent_id": 1,
                "child_id": 2,
                "child_wins": 55,
                "parent_wins": 40,
                "draws": 5,
                "total_games": 100,
                "win_rate": 0.55,
                "ci_lower": 0.45,
                "ci_upper": 0.65,
                "is_significant": false
            }
        }
        """
        try:
            data = await request.json()
            parent_id = data.get("parent_generation_id")
            child_id = data.get("child_generation_id")
            num_games = data.get("num_games", 100)
            harness_type = data.get("harness_type", "gumbel_mcts")
            difficulty = data.get("difficulty", 1.0)

            if not parent_id or not child_id:
                return self.error_response(
                    "parent_generation_id and child_generation_id required",
                    status=400
                )

            # Load generation tracker
            try:
                from app.coordination.generation_tracker import get_generation_tracker
                tracker = get_generation_tracker()
            except ImportError:
                return self.error_response(
                    "Generation tracker not available",
                    status=503
                )

            parent = tracker.get_generation(parent_id)
            child = tracker.get_generation(child_id)

            if not parent or not child:
                return self.error_response(
                    f"Generation not found: parent={parent_id}, child={child_id}",
                    status=404
                )

            # Verify model paths exist
            from pathlib import Path
            if not parent.model_path or not Path(parent.model_path).exists():
                return self.error_response(
                    f"Parent model not found: {parent.model_path}",
                    status=404
                )
            if not child.model_path or not Path(child.model_path).exists():
                return self.error_response(
                    f"Child model not found: {child.model_path}",
                    status=404
                )

            logger.info(
                f"Starting generation tournament: Gen {child_id} vs Gen {parent_id} "
                f"({num_games} games, harness={harness_type})"
            )

            # Run the tournament
            try:
                from app.training.tournament import run_tournament
                from app.models import BoardType

                board_type_map = {
                    "hex8": BoardType.HEX8,
                    "square8": BoardType.SQUARE8,
                    "square19": BoardType.SQUARE19,
                    "hexagonal": BoardType.HEXAGONAL,
                }
                board_type = board_type_map.get(child.board_type, BoardType.HEX8)

                results = await asyncio.to_thread(
                    run_tournament,
                    model_a_path=child.model_path,
                    model_b_path=parent.model_path,
                    num_games=num_games,
                    board_type=board_type,
                    num_players=child.num_players,
                )

                child_wins = results["model_a_wins"]
                parent_wins = results["model_b_wins"]
                draws = results.get("draws", 0)
                total = child_wins + parent_wins + draws

                # Calculate Wilson CI
                from app.training.significance import wilson_score_interval
                win_rate = child_wins / total if total > 0 else 0.0
                ci_lower, ci_upper = wilson_score_interval(child_wins, total, confidence=0.95)
                is_significant = ci_lower > 0.5

                # Record tournament result
                tracker.record_tournament(
                    gen_a=child_id,
                    gen_b=parent_id,
                    gen_a_wins=child_wins,
                    gen_b_wins=parent_wins,
                    draws=draws,
                )

                stats = {
                    "parent_id": parent_id,
                    "child_id": child_id,
                    "child_wins": child_wins,
                    "parent_wins": parent_wins,
                    "draws": draws,
                    "total_games": total,
                    "win_rate": win_rate,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "is_significant": is_significant,
                }

                logger.info(
                    f"Generation tournament completed: Gen {child_id} {child_wins}/{total} "
                    f"({win_rate:.1%}), significant={is_significant}"
                )

                return self.json_response({
                    "status": "completed",
                    "stats": stats,
                })

            except ImportError as e:
                return self.error_response(f"Tournament module not available: {e}", status=503)
            except Exception as e:
                logger.error(f"Tournament execution failed: {e}")
                return self.error_response(f"Tournament failed: {e}", status=500)

        except Exception as e:
            logger.exception(f"Generation tournament handler error: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_diverse_tournament(self, request: web.Request) -> web.Response:
        """Execute tournament against diverse opponents.

        Tests a model against a diverse set of opponents including:
        - Random and heuristic baselines
        - NN with various harnesses (MCTS, policy-only, descent)
        - NNUE with appropriate search (minimax for 2p, maxn/brs for 3-4p)

        Request body:
        {
            "model_path": "models/canonical_hex8_2p.pth",
            "board_type": "hex8",
            "num_players": 2,
            "diversity_level": "standard",
            "games_per_opponent": 20
        }

        Response:
        {
            "status": "completed",
            "model_path": "...",
            "results": [
                {"opponent": "random_baseline", "wins": 19, "losses": 1, "draws": 0},
                {"opponent": "heuristic_strong", "wins": 12, "losses": 8, "draws": 0},
                ...
            ],
            "summary": {
                "total_games": 200,
                "total_wins": 140,
                "overall_win_rate": 0.70
            }
        }
        """
        try:
            data = await request.json()
            model_path = data.get("model_path")
            board_type = data.get("board_type")
            num_players = data.get("num_players", 2)
            diversity_level = data.get("diversity_level", "standard")
            games_per_opponent = data.get("games_per_opponent", 20)

            if not model_path or not board_type:
                return self.error_response(
                    "model_path and board_type required",
                    status=400
                )

            # Check model exists
            from pathlib import Path
            if not Path(model_path).exists():
                return self.error_response(f"Model not found: {model_path}", status=404)

            logger.info(
                f"Starting diverse tournament for {model_path} "
                f"({board_type}_{num_players}p, diversity={diversity_level})"
            )

            # Get diverse opponents
            try:
                from app.coordination.tournament_daemon import OpponentDiversityMatrix
                matrix = OpponentDiversityMatrix()
                opponents = matrix.get_opponents_for_config(
                    board_type=board_type,
                    num_players=num_players,
                    diversity_level=diversity_level,
                    canonical_model_path=model_path,
                )
            except ImportError as e:
                return self.error_response(f"Diversity matrix not available: {e}", status=503)

            # Run tournaments against each opponent
            results = []
            total_wins = 0
            total_games = 0

            for opponent in opponents:
                try:
                    result = await self._run_diverse_opponent_match(
                        model_path=model_path,
                        opponent=opponent,
                        board_type=board_type,
                        num_players=num_players,
                        num_games=games_per_opponent,
                    )
                    results.append(result)
                    total_wins += result.get("wins", 0)
                    total_games += result.get("wins", 0) + result.get("losses", 0) + result.get("draws", 0)
                except Exception as e:
                    logger.warning(f"Match against {opponent.name} failed: {e}")
                    results.append({
                        "opponent": opponent.name,
                        "error": str(e),
                    })

            overall_win_rate = total_wins / total_games if total_games > 0 else 0.0

            logger.info(
                f"Diverse tournament completed: {len(results)} opponents, "
                f"{total_wins}/{total_games} wins ({overall_win_rate:.1%})"
            )

            return self.json_response({
                "status": "completed",
                "model_path": model_path,
                "results": results,
                "summary": {
                    "total_games": total_games,
                    "total_wins": total_wins,
                    "overall_win_rate": overall_win_rate,
                    "opponents_tested": len(results),
                },
            })

        except Exception as e:
            logger.exception(f"Diverse tournament handler error: {e}")
            return self.error_response(str(e), status=500)

    async def _run_diverse_opponent_match(
        self,
        model_path: str,
        opponent: Any,
        board_type: str,
        num_players: int,
        num_games: int,
    ) -> dict:
        """Run games against a single diverse opponent.

        Args:
            model_path: Path to the model being evaluated.
            opponent: OpponentSpec defining the opponent configuration.
            board_type: Board type string.
            num_players: Number of players.
            num_games: Number of games to play.

        Returns:
            Dict with opponent name, wins, losses, draws.
        """
        from app.ai.harness.harness_registry import create_harness_with_difficulty
        from app.ai.harness.base_harness import HarnessType, ModelType
        from app.models import BoardType

        board_type_map = {
            "hex8": BoardType.HEX8,
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hexagonal": BoardType.HEXAGONAL,
        }
        bt = board_type_map.get(board_type, BoardType.HEX8)

        # Create harness for opponent
        opponent_harness = create_harness_with_difficulty(
            harness_type=opponent.harness_type,
            difficulty=opponent.difficulty,
            model_path=opponent.model_id,
            board_type=bt,
            num_players=num_players,
            model_type=opponent.model_type,
        )

        # Create harness for model being evaluated (standard settings)
        model_harness = create_harness_with_difficulty(
            harness_type=HarnessType.GUMBEL_MCTS,
            difficulty=1.0,
            model_path=model_path,
            board_type=bt,
            num_players=num_players,
            model_type=ModelType.NEURAL_NET,
        )

        # Run games
        wins = 0
        losses = 0
        draws = 0

        for game_idx in range(num_games):
            try:
                # Alternate who goes first
                if game_idx % 2 == 0:
                    result = await asyncio.to_thread(
                        self._play_harness_match,
                        model_harness, opponent_harness, bt, num_players
                    )
                else:
                    result = await asyncio.to_thread(
                        self._play_harness_match,
                        opponent_harness, model_harness, bt, num_players
                    )
                    # Flip result since opponent went first
                    result = -result if result != 0 else 0

                if result > 0:
                    wins += 1
                elif result < 0:
                    losses += 1
                else:
                    draws += 1
            except Exception as e:
                logger.warning(f"Game {game_idx} failed: {e}")

        return {
            "opponent": opponent.name,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / num_games if num_games > 0 else 0.0,
        }

    def _play_harness_match(
        self,
        harness_a: Any,
        harness_b: Any,
        board_type: Any,
        num_players: int,
    ) -> int:
        """Play a single game between two harnesses.

        Returns:
            1 if harness_a wins, -1 if harness_b wins, 0 for draw.
        """
        from app.rules.engine import GameEngine
        from app.rules.ring_rift_state import RingRiftState

        # Initialize game state
        state = RingRiftState.create_initial_state(
            board_type=board_type,
            num_players=num_players,
        )

        harnesses = [harness_a, harness_b]
        current_player = 0
        max_moves = 500  # Safety limit

        for _ in range(max_moves):
            if state.is_terminal():
                break

            harness = harnesses[current_player % 2]
            move = harness.select_move(state)

            if move is None:
                break

            state = GameEngine.apply_move(state, move)
            current_player += 1

        # Determine winner
        if state.winner is not None:
            if state.winner == 0:
                return 1  # harness_a wins
            else:
                return -1  # harness_b wins

        return 0  # Draw or incomplete
