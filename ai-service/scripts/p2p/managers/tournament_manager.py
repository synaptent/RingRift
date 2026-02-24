"""Tournament Manager for P2P Orchestrator.

January 2026: Phase 11 Aggressive Decomposition - Extracts tournament
coordination, match execution, and gossip-based scheduling from
the monolithic p2p_orchestrator.py.

This manager handles:
- Model comparison tournaments after training
- Distributed tournament scheduling via gossip
- Tournament consensus and coordinator election
- Match execution and result processing
- Selfplay boost after model promotion

Dependencies:
- Orchestrator reference for peers, job_manager, notifier
- DistributedTournamentState, SSHTournamentRun from models
- Elo database for rating updates
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.async_context import safe_create_task

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator

logger = logging.getLogger(__name__)

# Singleton instance
_tournament_manager: "TournamentManager | None" = None


@dataclass
class TournamentConfig:
    """Configuration for TournamentManager."""

    match_timeout: float = 300.0  # 5 minutes per match
    tournament_timeout: float = 3600.0  # 1 hour per tournament
    games_per_comparison: int = 50
    promotion_win_rate_threshold: float = 0.55
    consensus_timeout: float = 600.0  # 10 minutes for proposal expiry
    max_workers_per_tournament: int = 10
    selfplay_boost_duration: float = 3600.0  # 1 hour boost after promotion
    selfplay_boost_multiplier: float = 1.5


@dataclass
class TournamentStats:
    """Statistics tracked by TournamentManager."""

    matches_played: int = 0
    matches_timed_out: int = 0
    matches_errored: int = 0
    tournaments_scheduled: int = 0
    tournaments_completed: int = 0
    tournaments_failed: int = 0
    models_promoted: int = 0
    promotions_rejected: int = 0
    gossip_proposals_received: int = 0
    consensus_reached: int = 0


class TournamentManager:
    """Manages tournament coordination and match execution.

    Extracted from P2POrchestrator in January 2026 (Phase 11) to improve
    modularity and testability.
    """

    def __init__(
        self,
        config: TournamentConfig | None = None,
        orchestrator: "P2POrchestrator | None" = None,
    ):
        self.config = config or TournamentConfig()
        self._orchestrator = orchestrator
        self._stats = TournamentStats()

        # Distributed tournament scheduling state
        self._tournament_proposals: dict[str, dict] = {}
        self._tournament_votes: dict[str, dict[str, str]] = {}
        self._active_tournaments_gossip: dict[str, dict] = {}
        self._last_tournament_check: float = 0.0
        self._tournament_coordination_lock = threading.Lock()

        # Selfplay boost tracking
        self._selfplay_boost_configs: dict[str, dict] = {}

    # =========================================================================
    # Core Tournament Methods
    # =========================================================================

    async def play_tournament_match(
        self, job_id: str, match_info: dict
    ) -> dict | None:
        """Play a tournament match locally using subprocess selfplay.

        Dec 28, 2025: Fixed to accept both field name conventions:
        - agent1/agent2 (from tournament handler)
        - player1_model/player2_model (from JobManager)

        Returns:
            Match result dict or None on error
        """
        if not self._orchestrator:
            logger.warning("TournamentManager: No orchestrator reference")
            return None

        try:
            import json as json_module

            # Support both naming conventions
            agent1 = match_info.get("agent1") or match_info.get("player1_model")
            agent2 = match_info.get("agent2") or match_info.get("player2_model")
            game_num = match_info.get("game_num", 0)
            board_type = match_info.get("board_type", "square8")
            num_players = match_info.get("num_players", 2)

            logger.info(f"Playing tournament match: {agent1} vs {agent2} (game {game_num})")

            # Build the subprocess command to run a single game
            ai_service_path = self._orchestrator._get_ai_service_path()
            game_script = f"""
import sys
sys.path.insert(0, '{ai_service_path}')
from app.training.initial_state import create_initial_state
from app.rules import get_rules_engine
from app.ai.heuristic_ai import HeuristicAI
from app.ai.random_ai import RandomAI
from app.models import AIConfig
import json
import os

# Skip shadow contracts for performance
os.environ['RINGRIFT_SKIP_SHADOW_CONTRACTS'] = 'true'

def load_agent(agent_id: str, player_idx: int, board_type: str, num_players: int, game_seed: int = 0):
    '''Load agent by ID - supports random, heuristic, or model paths.'''
    rng_seed = (game_seed * 10000 + player_idx * 1000) & 0xFFFFFFFF
    config = AIConfig(board_type=board_type, num_players=num_players, difficulty=5, rng_seed=rng_seed)
    if agent_id == 'random':
        return RandomAI(player_idx, config=config)
    elif agent_id == 'heuristic':
        return HeuristicAI(player_idx, config=config)
    elif agent_id.startswith('heuristic:'):
        weight_str = agent_id.split(':')[1]
        try:
            weights = [float(w) for w in weight_str.split(',')]
            weight_names = [
                "material_weight", "ring_count_weight", "stack_height_weight",
                "center_control_weight", "territory_weight", "mobility_weight",
                "line_potential_weight", "defensive_weight",
            ]
            weight_dict = dict(zip(weight_names, weights))
            config.heuristic_weights = weight_dict
        except (ValueError, TypeError):
            pass  # Fall through with default heuristic weights
        return HeuristicAI(player_idx, config=config)
    elif agent_id.startswith('model:') or agent_id.startswith('canonical_'):
        return HeuristicAI(player_idx, config=config)
    else:
        return HeuristicAI(player_idx, config=config)

engine = get_rules_engine(skip_shadow_contracts=True)
state = create_initial_state(board_type='{board_type}', num_players={num_players})
agents = [
    load_agent('{agent1}', 0, '{board_type}', {num_players}, {game_num}),
    load_agent('{agent2}', 1, '{board_type}', {num_players}, {game_num}),
]

max_moves = 10000
move_count = 0
while not state.game_over and move_count < max_moves:
    current_player = state.current_player_index
    agent = agents[current_player]
    move = agent.select_move(state)
    if move is None:
        break
    state = engine.apply_move(state, move)
    move_count += 1

winner_idx = None
victory_type = 'unknown'
if state.game_over:
    scores = state.player_scores
    if scores:
        max_score = max(scores)
        if scores.count(max_score) == 1:
            winner_idx = scores.index(max_score)

winner_agent = None
if winner_idx == 0:
    winner_agent = '{agent1}'
elif winner_idx == 1:
    winner_agent = '{agent2}'

result = {{
    'agent1': '{agent1}',
    'agent2': '{agent2}',
    'winner': winner_agent,
    'winner_idx': winner_idx,
    'victory_type': victory_type,
    'move_count': move_count,
    'game_num': {game_num},
}}
print(json.dumps(result))
"""
            # Run the game in subprocess
            cmd = [sys.executable, "-c", game_script]
            env = os.environ.copy()
            env["PYTHONPATH"] = ai_service_path
            env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.match_timeout
            )

            if proc.returncode != 0:
                logger.info(f"Tournament match subprocess error: {stderr.decode()}")
                self._stats.matches_errored += 1
                result = {
                    "agent1": agent1,
                    "agent2": agent2,
                    "winner": None,
                    "error": stderr.decode()[:200],
                    "game_num": game_num,
                }
            else:
                output_lines = stdout.decode().strip().split('\n')
                result_line = output_lines[-1] if output_lines else '{}'
                result = json_module.loads(result_line)
                self._stats.matches_played += 1

            logger.info(f"Match result: {agent1} vs {agent2} -> winner={result.get('winner')}")

            # Report result back to coordinator (leader)
            await self._report_match_result(job_id, result)

            return result

        except asyncio.TimeoutError:
            logger.info(f"Tournament match timed out: {match_info}")
            self._stats.matches_timed_out += 1
            return None
        except Exception as e:  # noqa: BLE001
            logger.info(f"Tournament match error: {e}")
            self._stats.matches_errored += 1
            return None

    async def _report_match_result(self, job_id: str, result: dict) -> None:
        """Report match result back to coordinator."""
        if not self._orchestrator:
            return

        try:
            from scripts.p2p.models import NodeRole
            from scripts.p2p.http_utils import get_client_session
            from aiohttp import ClientTimeout

            if self._orchestrator.role != NodeRole.LEADER and self._orchestrator.leader_id:
                with self._orchestrator.peers_lock:
                    leader = self._orchestrator.peers.get(self._orchestrator.leader_id)
                if leader:
                    try:
                        timeout = ClientTimeout(total=10)
                        async with get_client_session(timeout) as session:
                            url = self._orchestrator._url_for_peer(leader, "/tournament/result")
                            await session.post(url, json={
                                "job_id": job_id,
                                "result": result,
                                "worker_id": self._orchestrator.node_id,
                            }, headers=self._orchestrator._auth_headers())
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"Failed to report tournament result to leader: {e}")
            else:
                # We are the leader, update state directly
                if job_id in self._orchestrator.distributed_tournament_state:
                    state = self._orchestrator.distributed_tournament_state[job_id]
                    state.results.append(result)
                    state.completed_matches += 1
                    state.last_update = time.time()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Result reporting skipped: {e}")

    async def schedule_model_comparison(
        self, job: Any, new_model_path: str
    ) -> None:
        """Schedule a tournament to compare new model against current baseline.

        LEARNED LESSONS - After training, automatically run tournament to:
        1. Compare new model against current best baseline
        2. Update Elo ratings
        3. Promote to best baseline if new model wins
        """
        if not self._orchestrator:
            return

        try:
            from scripts.p2p.models import SSHTournamentRun

            config_key = f"{job.board_type}_{job.num_players}p"
            logger.info(f"Scheduling model comparison tournament for {config_key}")

            # Find current baseline model
            ai_service_path = self._orchestrator._get_ai_service_path()
            baseline_dir = Path(ai_service_path) / "models" / job.job_type
            baseline_pattern = f"{job.board_type}_{job.num_players}p_best*"

            baseline_model = None
            for f in baseline_dir.glob(baseline_pattern):
                baseline_model = str(f)
                break

            if not baseline_model:
                # No baseline - this model becomes baseline
                logger.info(f"No baseline found for {config_key}, new model becomes baseline")
                await self.promote_to_baseline(new_model_path, job.board_type, job.num_players, job.job_type)
                return

            # Schedule tournament via SSH tournament system
            tournament_id = f"autoeval_{config_key}_{int(time.time())}"

            with self._orchestrator.ssh_tournament_lock:
                self._orchestrator.ssh_tournament_runs[tournament_id] = SSHTournamentRun(
                    tournament_id=tournament_id,
                    board_type=job.board_type,
                    num_players=job.num_players,
                    status="pending",
                    started_at=time.time(),
                )

            # Start tournament in background
            tournament_config = {
                "tournament_id": tournament_id,
                "board_type": job.board_type,
                "num_players": job.num_players,
                "model_a": new_model_path,
                "model_b": baseline_model,
                "games_per_matchup": self.config.games_per_comparison,
            }
            safe_create_task(self.run_model_comparison_tournament(tournament_config), name="tournament-model-comparison")
            self._stats.tournaments_scheduled += 1

        except Exception as e:  # noqa: BLE001
            logger.info(f"Model comparison scheduling error: {e}")

    async def run_model_comparison_tournament(self, config: dict) -> None:
        """Run a model comparison tournament and update baseline if new model wins."""
        if not self._orchestrator:
            return

        tournament_id = config["tournament_id"]
        try:
            logger.info(f"Running model comparison tournament {tournament_id}")

            ai_service_path = self._orchestrator._get_ai_service_path()
            results_dir = Path(ai_service_path) / "results" / "tournaments"
            results_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                os.path.join(ai_service_path, "scripts", "run_tournament.py"),
                "--player1", f"nn:{config['model_a']}",
                "--player2", f"nn:{config['model_b']}",
                "--board", config["board_type"],
                "--num-players", str(config["num_players"]),
                "--games", str(config["games_per_matchup"]),
                "--output", str(results_dir / f"{tournament_id}.json"),
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = ai_service_path
            env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            _stdout, _stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.tournament_timeout
            )

            if proc.returncode == 0:
                results_file = results_dir / f"{tournament_id}.json"
                if results_file.exists():
                    import json as json_module
                    results = json_module.loads(results_file.read_text())
                    new_model_wins = results.get("player1_wins", 0)
                    baseline_wins = results.get("player2_wins", 0)
                    total_games = new_model_wins + baseline_wins

                    win_rate = new_model_wins / total_games if total_games > 0 else 0.5
                    logger.info(f"Tournament {tournament_id}: new model win rate = {win_rate:.1%}")

                    promoted = win_rate >= self.config.promotion_win_rate_threshold
                    if promoted:
                        logger.info("New model beats baseline! Promoting to best baseline.")
                        await self.promote_to_baseline(
                            config["model_a"], config["board_type"],
                            config["num_players"], "nnue" if "nnue" in config["model_a"].lower() else "cmaes"
                        )
                        self._stats.models_promoted += 1
                    else:
                        self._stats.promotions_rejected += 1

                    # Update improvement cycle manager with tournament result
                    await self.handle_tournament_completion(
                        tournament_id,
                        config["board_type"],
                        config["num_players"],
                        config["model_a"],
                        config["model_b"],
                        win_rate,
                        promoted,
                    )
                    self._stats.tournaments_completed += 1

            with self._orchestrator.ssh_tournament_lock:
                if tournament_id in self._orchestrator.ssh_tournament_runs:
                    self._orchestrator.ssh_tournament_runs[tournament_id].status = "completed"
                    self._orchestrator.ssh_tournament_runs[tournament_id].completed_at = time.time()

        except Exception as e:  # noqa: BLE001
            logger.info(f"Tournament {tournament_id} error: {e}")
            self._stats.tournaments_failed += 1
            if self._orchestrator:
                with self._orchestrator.ssh_tournament_lock:
                    if tournament_id in self._orchestrator.ssh_tournament_runs:
                        self._orchestrator.ssh_tournament_runs[tournament_id].status = "failed"
                        self._orchestrator.ssh_tournament_runs[tournament_id].error = str(e)

    async def handle_tournament_completion(
        self,
        tournament_id: str,
        board_type: str,
        num_players: int,
        new_model: str,
        baseline_model: str,
        win_rate: float,
        promoted: bool,
    ) -> None:
        """Handle tournament completion - update cycle state and trigger next iteration.

        This closes the feedback loop by:
        1. Updating improvement cycle manager with evaluation result
        2. Recording result to unified Elo database
        3. Updating diversity metrics
        4. Boosting selfplay for this config if model was promoted
        """
        if not self._orchestrator:
            return

        try:
            # 1. Update improvement cycle manager
            if self._orchestrator.improvement_cycle_manager:
                self._orchestrator.improvement_cycle_manager.handle_evaluation_complete(
                    board_type, num_players, win_rate, new_model
                )
                logger.info(f"Updated improvement cycle for {board_type}_{num_players}p")

            # 2. Record to unified Elo database
            try:
                from app.tournament import get_elo_database
                db = get_elo_database()
                rankings = [0, 1] if win_rate > 0.5 else [1, 0]
                db.record_match_and_update(
                    participant_ids=[new_model, baseline_model],
                    rankings=rankings,
                    board_type=board_type,
                    num_players=num_players,
                    tournament_id=tournament_id,
                )
                logger.info("Recorded tournament result to unified Elo DB")

                # Trigger Elo sync to propagate to cluster
                if hasattr(self._orchestrator, 'elo_sync_manager') and self._orchestrator.elo_sync_manager:
                    safe_create_task(self._orchestrator._trigger_elo_sync_after_matches(1), name="tournament-elo-sync")
            except Exception as e:  # noqa: BLE001
                logger.info(f"Elo database update failed (non-fatal): {e}")

            # 3. Update diversity metrics
            if hasattr(self._orchestrator, 'diversity_metrics'):
                self._orchestrator.diversity_metrics["tournament_runs"] = self._orchestrator.diversity_metrics.get("tournament_runs", 0) + 1
                if promoted:
                    self._orchestrator.diversity_metrics["promotions"] = self._orchestrator.diversity_metrics.get("promotions", 0) + 1

            # 4. Record metrics for observability
            self._orchestrator.record_metric(
                "tournament_win_rate",
                win_rate,
                board_type=board_type,
                num_players=num_players,
                metadata={
                    "new_model": new_model,
                    "baseline_model": baseline_model,
                    "promoted": promoted,
                    "tournament_id": tournament_id,
                },
            )

            # 5. Boost selfplay for this config if promoted
            if promoted:
                safe_create_task(self.boost_selfplay_for_config(board_type, num_players), name="tournament-boost-selfplay")
                # Alert on successful promotion
                safe_create_task(self._orchestrator.notifier.send(
                    title="Model Promoted",
                    message=f"New model promoted for {board_type}_{num_players}p with {win_rate*100:.1f}% win rate",
                    level="info",
                    fields={"Model": new_model, "Win Rate": f"{win_rate*100:.1f}%"},
                    node_id=self._orchestrator.node_id,
                ), name="tournament-notify-promoted")
            elif win_rate < 0.5:
                # Alert on failed promotion
                safe_create_task(self._orchestrator.notifier.send(
                    title="Model Promotion Failed",
                    message=f"New model failed tournament for {board_type}_{num_players}p with only {win_rate*100:.1f}% win rate",
                    level="warning",
                    fields={
                        "Model": new_model,
                        "Win Rate": f"{win_rate*100:.1f}%",
                        "Baseline": baseline_model,
                    },
                    node_id=self._orchestrator.node_id,
                ), name="tournament-notify-failed")

        except Exception as e:  # noqa: BLE001
            logger.info(f"Tournament completion handler error: {e}")
            if self._orchestrator and self._orchestrator.notifier:
                safe_create_task(self._orchestrator.notifier.send(
                    title="Tournament Handler Error",
                    message=str(e),
                    level="error",
                    node_id=self._orchestrator.node_id,
                ), name="tournament-notify-error")

    async def promote_to_baseline(
        self, model_path: str, board_type: str, num_players: int, model_type: str
    ) -> None:
        """Promote a model to the best baseline for its board type."""
        if not self._orchestrator:
            return

        try:
            ai_service_path = self._orchestrator._get_ai_service_path()
            baseline_dir = Path(ai_service_path) / "models" / model_type
            baseline_dir.mkdir(parents=True, exist_ok=True)

            baseline_path = baseline_dir / f"{board_type}_{num_players}p_best.pt"
            if baseline_path.exists():
                backup_path = baseline_dir / f"{board_type}_{num_players}p_prev_{int(time.time())}.pt"
                shutil.copy2(baseline_path, backup_path)
                logger.info(f"Backed up previous baseline to {backup_path}")

            shutil.copy2(model_path, baseline_path)
            logger.info(f"Promoted {model_path} to baseline at {baseline_path}")

            # Emit MODEL_PROMOTED event for coordination layer
            config_key = f"{board_type}_{num_players}p"
            model_id = Path(model_path).name
            await self._orchestrator._emit_model_promoted(
                model_id=model_id,
                config_key=config_key,
                elo=0.0,
                elo_gain=0.0,
                source="tournament_manager.promote_to_baseline",
            )

        except Exception as e:  # noqa: BLE001
            logger.info(f"Baseline promotion error: {e}")

    async def boost_selfplay_for_config(self, board_type: str, num_players: int) -> None:
        """Temporarily boost selfplay for a configuration after model promotion.

        This accelerates data generation for the next training iteration.
        """
        try:
            config_key = f"{board_type}_{num_players}p"
            logger.info(f"Boosting selfplay for {config_key} after promotion")

            self._selfplay_boost_configs[config_key] = {
                "boost_until": time.time() + self.config.selfplay_boost_duration,
                "multiplier": self.config.selfplay_boost_multiplier,
            }

            # Also update orchestrator if it has boost tracking
            if self._orchestrator and hasattr(self._orchestrator, 'selfplay_boost_configs'):
                self._orchestrator.selfplay_boost_configs[config_key] = self._selfplay_boost_configs[config_key]

        except Exception as e:  # noqa: BLE001
            logger.info(f"Selfplay boost error: {e}")

    def get_selfplay_boost(self, config_key: str) -> float:
        """Get current selfplay boost multiplier for a config."""
        now = time.time()
        boost_info = self._selfplay_boost_configs.get(config_key, {})
        if boost_info and now < boost_info.get("boost_until", 0):
            return boost_info.get("multiplier", 1.0)
        return 1.0

    # =========================================================================
    # Distributed Tournament Scheduling (Gossip-based)
    # =========================================================================

    def init_distributed_tournament_scheduling(self) -> None:
        """Initialize distributed tournament scheduling state."""
        self._tournament_proposals = {}
        self._tournament_votes = {}
        self._active_tournaments_gossip = {}
        self._last_tournament_check = 0
        # Lock already initialized in __init__

    def get_tournament_gossip_state(self) -> dict:
        """Get tournament state for gossip propagation.

        TOURNAMENT GOSSIP: Share active tournament info via gossip so nodes
        can coordinate without leader.

        Returns:
            Dict with proposals and active tournaments
        """
        if not self._orchestrator:
            return {"proposals": [], "active": {}, "last_update": time.time()}

        now = time.time()

        with self._tournament_coordination_lock:
            # Only share recent proposals (last 10 min)
            active_proposals = {
                pid: p for pid, p in self._tournament_proposals.items()
                if now - p.get("proposed_at", 0) < self.config.consensus_timeout
                and p.get("status") == "proposed"
            }

        # Get active distributed tournaments
        active_tournaments = {}
        distributed_state = getattr(self._orchestrator, "distributed_tournament_state", {})
        for tid, state in distributed_state.items():
            if hasattr(state, "status") and state.status == "running":
                active_tournaments[tid] = {
                    "job_id": tid,
                    "coordinator": self._orchestrator.node_id,
                    "progress": state.completed_matches / max(1, state.total_matches),
                    "status": state.status,
                }

        return {
            "proposals": list(active_proposals.values()),
            "active": active_tournaments,
            "last_update": now,
        }

    def process_tournament_gossip(self, node_id: str, tournament_state: dict) -> None:
        """Process tournament info received via gossip.

        GOSSIP PROCESSING: When receiving tournament state from peers,
        - Record their proposals and votes
        - Check if any proposals reached consensus
        - Start tournaments that we're elected to coordinate
        """
        if not self._orchestrator:
            return

        if not tournament_state or not isinstance(tournament_state, dict):
            return

        self._stats.gossip_proposals_received += 1

        # Process proposals from gossip
        for proposal in tournament_state.get("proposals", []):
            if not isinstance(proposal, dict):
                continue

            proposal_id = proposal.get("proposal_id")
            if not proposal_id:
                continue

            with self._tournament_coordination_lock:
                if proposal_id not in self._tournament_proposals:
                    # New proposal from peer - add it and auto-approve
                    self._tournament_proposals[proposal_id] = proposal.copy()
                    if "votes" not in self._tournament_proposals[proposal_id]:
                        self._tournament_proposals[proposal_id]["votes"] = {}
                    self._tournament_proposals[proposal_id]["votes"][self._orchestrator.node_id] = "approve"
                else:
                    # Merge votes
                    existing = self._tournament_proposals[proposal_id]
                    for voter, vote in proposal.get("votes", {}).items():
                        if voter not in existing.get("votes", {}):
                            if "votes" not in existing:
                                existing["votes"] = {}
                            existing["votes"][voter] = vote

    def check_tournament_consensus(self) -> None:
        """Check if any tournament proposals have reached consensus.

        CONSENSUS CHECK: A proposal is approved when majority of alive peers approve.
        The coordinator is elected as the highest-ID approving voter node.
        """
        if not self._orchestrator:
            return

        now = time.time()

        # Get alive peer count for quorum
        with self._orchestrator.peers_lock:
            alive_peers = [p for p in self._orchestrator.peers.values() if p.is_alive()]
        alive_count = len(alive_peers) + 1  # +1 for self

        quorum = (alive_count // 2) + 1

        with self._tournament_coordination_lock:
            for proposal_id, proposal in list(self._tournament_proposals.items()):
                if proposal.get("status") != "proposed":
                    continue

                # Count votes
                approve_votes = [
                    voter for voter, vote in proposal.get("votes", {}).items()
                    if vote == "approve"
                ]

                if len(approve_votes) >= quorum:
                    # Consensus reached! Elect coordinator (highest ID)
                    coordinator = max(approve_votes)
                    proposal["status"] = "approved"
                    proposal["coordinator"] = coordinator
                    self._stats.consensus_reached += 1

                    logger.info(f"TOURNAMENT: Proposal {proposal_id} approved! "
                          f"Coordinator: {coordinator} ({len(approve_votes)}/{alive_count} votes)")

                    # If we're the coordinator, start the tournament
                    if coordinator == self._orchestrator.node_id:
                        safe_create_task(self.start_tournament_from_proposal(proposal), name="tournament-from-proposal")

                # Expire old proposals
                elif now - proposal.get("proposed_at", 0) > self.config.consensus_timeout:
                    proposal["status"] = "expired"

    async def start_tournament_from_proposal(self, proposal: dict) -> None:
        """Start a tournament from an approved proposal.

        COORDINATOR DUTIES: When elected as coordinator, start the tournament
        and manage match distribution to workers.
        """
        if not self._orchestrator:
            return

        try:
            import uuid
            from scripts.p2p.models import DistributedTournamentState

            job_id = f"tournament_{uuid.uuid4().hex[:8]}"
            agent_ids = proposal.get("agent_ids", [])

            if len(agent_ids) < 2:
                logger.info("TOURNAMENT: Cannot start - need at least 2 agents")
                return

            # Create round-robin pairings
            pairings = []
            for i, a1 in enumerate(agent_ids):
                for a2 in agent_ids[i+1:]:
                    for game_num in range(proposal.get("games_per_pairing", 2)):
                        pairings.append({
                            "agent1": a1,
                            "agent2": a2,
                            "game_num": game_num,
                            "status": "pending",
                        })

            state = DistributedTournamentState(
                job_id=job_id,
                board_type=proposal.get("board_type", "square8"),
                num_players=proposal.get("num_players", 2),
                agent_ids=agent_ids,
                games_per_pairing=proposal.get("games_per_pairing", 2),
                total_matches=len(pairings),
                pending_matches=pairings,
                status="running",
                started_at=time.time(),
                last_update=time.time(),
            )

            # Find workers
            with self._orchestrator.peers_lock:
                workers = [p.node_id for p in self._orchestrator.peers.values() if p.is_healthy()]
            state.worker_nodes = workers

            if not state.worker_nodes:
                logger.info(f"TOURNAMENT: No workers available for {job_id}")
                return

            self._orchestrator.distributed_tournament_state[job_id] = state

            logger.info(f"TOURNAMENT: Started {job_id} from proposal {proposal.get('proposal_id')}: "
                  f"{len(agent_ids)} agents, {len(pairings)} matches, {len(workers)} workers")

            # Launch coordinator task
            safe_create_task(self._orchestrator.job_manager.run_distributed_tournament(job_id), name="tournament-distributed-run")

        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to start tournament from proposal: {e}")

    def get_distributed_tournament_summary(self) -> dict:
        """Get summary of distributed tournament scheduling for status endpoint."""
        with self._tournament_coordination_lock:
            pending_proposals = sum(
                1 for p in self._tournament_proposals.values()
                if p.get("status") == "proposed"
            )
            approved_proposals = sum(
                1 for p in self._tournament_proposals.values()
                if p.get("status") == "approved"
            )

        active_tournaments = 0
        if self._orchestrator:
            distributed_state = getattr(self._orchestrator, "distributed_tournament_state", {})
            active_tournaments = sum(
                1 for s in distributed_state.values()
                if hasattr(s, "status") and s.status == "running"
            )

        return {
            "pending_proposals": pending_proposals,
            "approved_proposals": approved_proposals,
            "active_tournaments": active_tournaments,
            "enabled": True,
        }

    # =========================================================================
    # Stats and Health
    # =========================================================================

    def get_stats(self) -> TournamentStats:
        """Get tournament statistics."""
        return self._stats

    def health_check(self) -> dict[str, Any]:
        """Return health check result for daemon protocol compliance."""
        return {
            "status": "healthy",
            "stats": {
                "matches_played": self._stats.matches_played,
                "tournaments_completed": self._stats.tournaments_completed,
                "models_promoted": self._stats.models_promoted,
                "consensus_reached": self._stats.consensus_reached,
            },
            "active_proposals": len([
                p for p in self._tournament_proposals.values()
                if p.get("status") == "proposed"
            ]),
            "boost_configs": list(self._selfplay_boost_configs.keys()),
        }


# =========================================================================
# Singleton Accessors
# =========================================================================

def create_tournament_manager(
    config: TournamentConfig | None = None,
    orchestrator: "P2POrchestrator | None" = None,
) -> TournamentManager:
    """Create a new TournamentManager instance."""
    return TournamentManager(config=config, orchestrator=orchestrator)


def get_tournament_manager() -> TournamentManager | None:
    """Get the singleton TournamentManager instance."""
    return _tournament_manager


def set_tournament_manager(manager: TournamentManager) -> None:
    """Set the singleton TournamentManager instance."""
    global _tournament_manager
    _tournament_manager = manager


def reset_tournament_manager() -> None:
    """Reset the singleton TournamentManager instance."""
    global _tournament_manager
    _tournament_manager = None
