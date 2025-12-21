#!/usr/bin/env python3
"""Run distributed Elo calibration tournament across P2P cluster.

This script leverages the P2P orchestrator network to run model-based Elo
calibration tournaments in parallel across all available nodes.

Features:
1. Discovers all P2P nodes via the orchestrator health endpoint
2. Distributes matches across nodes for parallel execution
3. Aggregates results into the central Elo leaderboard database
4. Supports both model-based and AI-type based tournaments

Usage:
    # Run calibration tournament for all AI types (for difficulty mapping)
    python scripts/run_p2p_elo_tournament.py --calibrate-ai-types --games 50

    # Run model Elo tournament across cluster
    python scripts/run_p2p_elo_tournament.py --models --games 100

    # Dry-run to see matchups without playing
    python scripts/run_p2p_elo_tournament.py --calibrate-ai-types --dry-run

    # Status of current tournament
    python scripts/run_p2p_elo_tournament.py --status
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sqlite3
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiohttp

# Add ai-service to path
AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.config.thresholds import ELO_K_FACTOR, INITIAL_ELO_RATING
from app.models import BoardType
from app.models.discovery import discover_models

# ============================================
# AI Type Calibration Configurations
# ============================================

# AI types to calibrate with their configurations
# These map to difficulty levels and are used to establish Elo baselines
AI_TYPE_CONFIGS = {
    "random": {
        "ai_type": "random",
        "difficulty": 1,
        "description": "Random legal moves",
    },
    "heuristic": {
        "ai_type": "heuristic",
        "difficulty": 2,
        "description": "Heuristic evaluation only",
    },
    "minimax_heuristic": {
        "ai_type": "minimax",
        "difficulty": 3,
        "use_neural_net": False,
        "description": "Minimax with heuristic eval",
    },
    "minimax_nnue": {
        "ai_type": "minimax",
        "difficulty": 4,
        "use_neural_net": True,
        "description": "Minimax with NNUE eval",
    },
    "mcts_heuristic": {
        "ai_type": "mcts",
        "difficulty": 5,
        "use_neural_net": False,
        "mcts_iterations": 200,
        "description": "MCTS with heuristic rollouts",
    },
    "mcts_neural": {
        "ai_type": "mcts",
        "difficulty": 6,
        "use_neural_net": True,
        "mcts_iterations": 400,
        "description": "MCTS with neural guidance",
    },
    "mcts_neural_high": {
        "ai_type": "mcts",
        "difficulty": 7,
        "use_neural_net": True,
        "mcts_iterations": 800,
        "description": "MCTS neural with higher budget",
    },
    "policy_only": {
        "ai_type": "policy_only",
        "difficulty": 3,  # TBD - calibrate this
        "policy_temperature": 0.5,
        "description": "Direct NN policy (no search)",
    },
    "gumbel_mcts": {
        "ai_type": "gumbel_mcts",
        "difficulty": 7,  # TBD - calibrate this
        "gumbel_num_sampled_actions": 16,
        "gumbel_simulation_budget": 100,
        "description": "Gumbel MCTS with Sequential Halving",
    },
    "descent": {
        "ai_type": "descent",
        "difficulty": 9,
        "description": "Policy Descent search",
    },
}


@dataclass
class P2PNode:
    """Information about a P2P node."""
    node_id: str
    host: str
    port: int
    has_gpu: bool = False
    gpu_name: str = ""
    selfplay_jobs: int = 0
    load_score: float = 0.0
    is_healthy: bool = True

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class MatchResult:
    """Result of a single match."""
    match_id: str
    agent_a: str
    agent_b: str
    winner: str  # "agent_a", "agent_b", or "draw"
    game_length: int
    duration_sec: float
    worker_node: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TournamentState:
    """State of the distributed tournament."""
    tournament_id: str
    board_type: str = "square8"
    num_players: int = 2
    agents: list[str] = field(default_factory=list)
    games_per_pairing: int = 2
    total_matches: int = 0
    completed_matches: int = 0
    pending_matches: list[tuple[str, str]] = field(default_factory=list)
    results: list[MatchResult] = field(default_factory=list)
    ratings: dict[str, float] = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed
    started_at: float = 0.0
    completed_at: float = 0.0
    error_message: str = ""


class P2PEloTournament:
    """Distributed Elo tournament coordinator."""

    def __init__(
        self,
        leader_host: str = "localhost",
        leader_port: int = 8770,
        board_type: str = "square8",
        num_players: int = 2,
        games_per_pairing: int = 2,
        elo_db_path: Path | None = None,
    ):
        self.leader_host = leader_host
        self.leader_port = leader_port
        self.board_type = board_type
        self.num_players = num_players
        self.games_per_pairing = games_per_pairing
        self.nodes: list[P2PNode] = []
        self.state: TournamentState | None = None
        self.elo_db_path = elo_db_path or (AI_SERVICE_ROOT / "data" / "elo_leaderboard.db")
        self._elo_service = None

    def _get_elo_service(self):
        if self._elo_service is None:
            try:
                from app.training.elo_service import get_elo_service
                self._elo_service = get_elo_service()
            except Exception as e:
                print(f"[Tournament] EloService unavailable: {e}")
                self._elo_service = None
        return self._elo_service

    def _register_participant(self, elo_service, agent_id: str) -> None:
        config = AI_TYPE_CONFIGS.get(agent_id, {})
        try:
            elo_service.register_participant(
                participant_id=agent_id,
                name=agent_id,
                ai_type=str(config.get("ai_type", agent_id)),
                difficulty=config.get("difficulty"),
                use_neural_net=bool(config.get("use_neural_net", False)),
                metadata={"source": "p2p_tournament", "description": config.get("description")},
            )
        except Exception as e:
            print(f"[Tournament] Failed to register participant {agent_id}: {e}")

    def _record_results_in_elo_service(self, results: list[MatchResult]) -> None:
        elo_service = self._get_elo_service()
        if elo_service is None:
            return

        tournament_id = self.state.tournament_id if self.state else "p2p_elo_tournament"
        for result in results:
            if result.winner == "agent_a":
                winner_id = result.agent_a
            elif result.winner == "agent_b":
                winner_id = result.agent_b
            else:
                winner_id = None

            self._register_participant(elo_service, result.agent_a)
            self._register_participant(elo_service, result.agent_b)

            try:
                elo_service.record_match(
                    participant_a=result.agent_a,
                    participant_b=result.agent_b,
                    winner=winner_id,
                    board_type=self.board_type,
                    num_players=self.num_players,
                    game_length=result.game_length,
                    duration_sec=result.duration_sec,
                    tournament_id=tournament_id,
                )
            except Exception as e:
                print(f"[Tournament] EloService record failed for {result.match_id}: {e}")

    async def discover_nodes(self) -> list[P2PNode]:
        """Discover all healthy P2P nodes via the leader."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                url = f"http://{self.leader_host}:{self.leader_port}/status"
                async with session.get(url) as resp:
                    if resp.status != 200:
                        print(f"[Tournament] Failed to get cluster status: {resp.status}")
                        return []

                    data = await resp.json()
                    peers = data.get("peers", {})

                    nodes = []
                    for node_id, info in peers.items():
                        if not info.get("is_alive", False):
                            continue

                        node = P2PNode(
                            node_id=node_id,
                            host=info.get("host", ""),
                            port=info.get("port", 8770),
                            has_gpu=info.get("has_gpu", False),
                            gpu_name=info.get("gpu_name", ""),
                            selfplay_jobs=info.get("selfplay_jobs", 0),
                            load_score=info.get("load_score", 0.0),
                            is_healthy=info.get("is_healthy", True),
                        )
                        nodes.append(node)

                    # Add leader node
                    leader_id = data.get("node_id", "leader")
                    leader_node = P2PNode(
                        node_id=leader_id,
                        host=self.leader_host,
                        port=self.leader_port,
                        has_gpu=data.get("gpu", {}).get("has_cuda", False),
                        gpu_name=data.get("gpu", {}).get("gpu_name", ""),
                        is_healthy=True,
                    )
                    nodes.append(leader_node)

                    self.nodes = nodes
                    return nodes

        except Exception as e:
            print(f"[Tournament] Error discovering nodes: {e}")
            return []

    def generate_round_robin_matchups(self, agents: list[str]) -> list[tuple[str, str]]:
        """Generate all pairings for round-robin tournament."""
        matchups = []
        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                # Play games_per_pairing games for each pairing
                for _ in range(self.games_per_pairing):
                    matchups.append((a, b))

        # Shuffle for fairness
        random.shuffle(matchups)
        return matchups

    async def run_match_on_node(
        self,
        node: P2PNode,
        agent_a: str,
        agent_b: str,
        match_id: str,
    ) -> MatchResult | None:
        """Run a single match on a remote node via P2P API."""
        try:
            # Use longer timeout for neural network matches
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes per match
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Use the enhanced Elo match endpoint
                url = f"{node.base_url}/tournament/play_elo_match"
                payload = {
                    "match_id": match_id,
                    "agent_a": agent_a,
                    "agent_b": agent_b,
                    "agent_a_config": AI_TYPE_CONFIGS.get(agent_a, {"ai_type": agent_a}),
                    "agent_b_config": AI_TYPE_CONFIGS.get(agent_b, {"ai_type": agent_b}),
                    "board_type": self.board_type,
                    "num_players": self.num_players,
                }

                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        print(f"[Tournament] Match failed on {node.node_id}: {resp.status} - {error_text[:100]}")
                        return None

                    result_data = await resp.json()
                    if result_data.get("success"):
                        return MatchResult(
                            match_id=match_id,
                            agent_a=agent_a,
                            agent_b=agent_b,
                            winner=result_data.get("winner", "draw"),
                            game_length=result_data.get("game_length", 0),
                            duration_sec=result_data.get("duration_sec", 0.0),
                            worker_node=result_data.get("worker_node", node.node_id),
                        )
                    else:
                        print(f"[Tournament] Match on {node.node_id} returned error: {result_data.get('error', 'unknown')}")
                        return None

        except asyncio.TimeoutError:
            print(f"[Tournament] Match timeout on {node.node_id}")
            return None
        except Exception as e:
            print(f"[Tournament] Error running match on {node.node_id}: {e}")
            return None

    async def run_match_locally(
        self,
        agent_a: str,
        agent_b: str,
        match_id: str,
    ) -> MatchResult | None:
        """Run a single match locally (fallback when no nodes available)."""
        try:
            # Import here to avoid circular imports
            from scripts.run_model_elo_tournament import play_model_vs_model_game

            config_a = AI_TYPE_CONFIGS.get(agent_a, {"ai_type": agent_a})
            config_b = AI_TYPE_CONFIGS.get(agent_b, {"ai_type": agent_b})

            board_type = BoardType(self.board_type)

            start_time = time.time()
            result = play_model_vs_model_game(
                model_a=config_a,
                model_b=config_b,
                board_type=board_type,
                num_players=self.num_players,
                save_game_history=False,
            )
            duration = time.time() - start_time

            winner_map = {
                "model_a": "agent_a",
                "model_b": "agent_b",
                "draw": "draw",
            }

            return MatchResult(
                match_id=match_id,
                agent_a=agent_a,
                agent_b=agent_b,
                winner=winner_map.get(result.get("winner", "draw"), "draw"),
                game_length=result.get("game_length", 0),
                duration_sec=duration,
                worker_node="local",
            )

        except Exception as e:
            print(f"[Tournament] Error running local match: {e}")
            return None

    def calculate_elo_ratings(self, results: list[MatchResult]) -> dict[str, float]:
        """Calculate Elo ratings from match results."""
        k_factor = ELO_K_FACTOR
        initial_rating = INITIAL_ELO_RATING

        # Collect all unique agents
        agents: set[str] = set()
        for r in results:
            agents.add(r.agent_a)
            agents.add(r.agent_b)

        ratings = dict.fromkeys(agents, initial_rating)

        # Process each result in order
        for r in sorted(results, key=lambda x: x.timestamp):
            ra = ratings[r.agent_a]
            rb = ratings[r.agent_b]

            # Expected scores
            ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400))
            eb = 1.0 - ea

            # Actual scores
            if r.winner == "agent_a":
                sa, sb = 1.0, 0.0
            elif r.winner == "agent_b":
                sa, sb = 0.0, 1.0
            else:
                sa, sb = 0.5, 0.5

            # Update ratings
            ratings[r.agent_a] = ra + k_factor * (sa - ea)
            ratings[r.agent_b] = rb + k_factor * (sb - eb)

        return ratings

    def save_ratings_to_db(self, ratings: dict[str, float], results: list[MatchResult]):
        """Save Elo ratings to the leaderboard database."""
        self.elo_db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(str(self.elo_db_path)) as conn:
            cursor = conn.cursor()

            # Create tables if needed
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_type_ratings (
                    agent_id TEXT PRIMARY KEY,
                    rating REAL,
                    games_played INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    draws INTEGER,
                    last_updated TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_type_match_history (
                    match_id TEXT PRIMARY KEY,
                    agent_a TEXT,
                    agent_b TEXT,
                    winner TEXT,
                    game_length INTEGER,
                    duration_sec REAL,
                    worker_node TEXT,
                    timestamp REAL
                )
            """)

            # Update ratings
            for agent, rating in ratings.items():
                # Count wins/losses/draws
                wins = sum(1 for r in results if
                    (r.agent_a == agent and r.winner == "agent_a") or
                    (r.agent_b == agent and r.winner == "agent_b"))
                losses = sum(1 for r in results if
                    (r.agent_a == agent and r.winner == "agent_b") or
                    (r.agent_b == agent and r.winner == "agent_a"))
                draws = sum(1 for r in results if
                    (r.agent_a == agent or r.agent_b == agent) and r.winner == "draw")
                games = wins + losses + draws

                cursor.execute("""
                    INSERT OR REPLACE INTO ai_type_ratings
                    (agent_id, rating, games_played, wins, losses, draws, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (agent, rating, games, wins, losses, draws, datetime.now().isoformat()))

            # Save match history
            for r in results:
                cursor.execute("""
                    INSERT OR REPLACE INTO ai_type_match_history
                    (match_id, agent_a, agent_b, winner, game_length, duration_sec, worker_node, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (r.match_id, r.agent_a, r.agent_b, r.winner, r.game_length,
                      r.duration_sec, r.worker_node, r.timestamp))

            conn.commit()
            print(f"[Tournament] Saved {len(ratings)} ratings and {len(results)} matches to {self.elo_db_path}")

    async def run_tournament(
        self,
        agents: list[str],
        max_parallel: int = 10,
        use_distributed: bool = True,
    ) -> TournamentState:
        """Run the full tournament."""
        tournament_id = f"elo_calibration_{uuid.uuid4().hex[:8]}"

        self.state = TournamentState(
            tournament_id=tournament_id,
            board_type=self.board_type,
            num_players=self.num_players,
            agents=agents,
            games_per_pairing=self.games_per_pairing,
            status="running",
            started_at=time.time(),
        )

        # Generate matchups
        matchups = self.generate_round_robin_matchups(agents)
        self.state.total_matches = len(matchups)
        self.state.pending_matches = matchups

        print(f"[Tournament] Starting tournament {tournament_id}")
        print(f"[Tournament] {len(agents)} agents, {len(matchups)} matches")

        # Discover nodes if using distributed mode
        if use_distributed:
            await self.discover_nodes()
            print(f"[Tournament] Found {len(self.nodes)} P2P nodes")

        results: list[MatchResult] = []

        # Run matches in parallel
        if use_distributed and len(self.nodes) > 0:
            # Distribute across nodes
            tasks = []

            for i, (agent_a, agent_b) in enumerate(matchups):
                match_id = f"{tournament_id}_match_{i}"
                node = self.nodes[i % len(self.nodes)]

                task = self.run_match_on_node(node, agent_a, agent_b, match_id)
                tasks.append(task)

                # Limit concurrency
                if len(tasks) >= max_parallel:
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    for r in batch_results:
                        if isinstance(r, MatchResult):
                            results.append(r)
                            self.state.completed_matches += 1
                            print(f"[Tournament] {self.state.completed_matches}/{self.state.total_matches} - {r.agent_a} vs {r.agent_b}: {r.winner}")
                    tasks = []

            # Process remaining tasks
            if tasks:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in batch_results:
                    if isinstance(r, MatchResult):
                        results.append(r)
                        self.state.completed_matches += 1
                        print(f"[Tournament] {self.state.completed_matches}/{self.state.total_matches} - {r.agent_a} vs {r.agent_b}: {r.winner}")
        else:
            # Run locally with thread pool
            print("[Tournament] Running locally (no P2P nodes available)")

            def run_local_match(args):
                i, agent_a, agent_b = args
                match_id = f"{tournament_id}_match_{i}"
                return asyncio.run(self.run_match_locally(agent_a, agent_b, match_id))

            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                futures = [
                    executor.submit(run_local_match, (i, a, b))
                    for i, (a, b) in enumerate(matchups)
                ]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            self.state.completed_matches += 1
                            print(f"[Tournament] {self.state.completed_matches}/{self.state.total_matches} - {result.agent_a} vs {result.agent_b}: {result.winner}")
                    except Exception as e:
                        print(f"[Tournament] Match error: {e}")

        # Calculate final ratings
        self.state.results = results
        self.state.ratings = self.calculate_elo_ratings(results)
        self.state.status = "completed"
        self.state.completed_at = time.time()

        # Save to database
        self.save_ratings_to_db(self.state.ratings, results)
        self._record_results_in_elo_service(results)

        return self.state

    def print_leaderboard(self):
        """Print the current Elo leaderboard."""
        if not self.state or not self.state.ratings:
            # Try loading from DB
            if self.elo_db_path.exists():
                with sqlite3.connect(str(self.elo_db_path)) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT agent_id, rating, games_played, wins, losses, draws
                        FROM ai_type_ratings
                        ORDER BY rating DESC
                    """)
                    rows = cursor.fetchall()
                    if rows:
                        print("\n" + "=" * 70)
                        print("AI TYPE ELO LEADERBOARD")
                        print("=" * 70)
                        print(f"{'Rank':<6} {'Agent':<25} {'Elo':<8} {'Games':<8} {'W/L/D':<12}")
                        print("-" * 70)
                        for i, (agent, rating, games, wins, losses, draws) in enumerate(rows, 1):
                            wld = f"{wins}/{losses}/{draws}"
                            print(f"{i:<6} {agent:<25} {rating:>7.1f} {games:<8} {wld:<12}")
                        print("=" * 70)
                        return
            print("[Tournament] No ratings available. Run a tournament first.")
            return

        print("\n" + "=" * 70)
        print("TOURNAMENT RESULTS")
        print("=" * 70)
        sorted_ratings = sorted(self.state.ratings.items(), key=lambda x: x[1], reverse=True)

        print(f"{'Rank':<6} {'Agent':<25} {'Elo':<8} {'Description':<30}")
        print("-" * 70)
        for i, (agent, rating) in enumerate(sorted_ratings, 1):
            desc = AI_TYPE_CONFIGS.get(agent, {}).get("description", "")
            print(f"{i:<6} {agent:<25} {rating:>7.1f} {desc:<30}")
        print("=" * 70)

        duration = self.state.completed_at - self.state.started_at
        print(f"\nCompleted {self.state.completed_matches} matches in {duration:.1f}s")
        print(f"Average: {duration / max(1, self.state.completed_matches):.2f}s per match")


async def main():
    parser = argparse.ArgumentParser(description="Run distributed Elo calibration tournament")
    parser.add_argument("--calibrate-ai-types", action="store_true",
                       help="Run AI type calibration tournament")
    parser.add_argument("--models", action="store_true",
                       help="Run model-based Elo tournament")
    parser.add_argument("--games", type=int, default=4,
                       help="Games per pairing (default: 4)")
    parser.add_argument("--board", type=str, default="square8",
                       choices=["square8", "square19", "hexagonal"],
                       help="Board type (default: square8)")
    parser.add_argument("--players", type=int, default=2,
                       help="Number of players (default: 2)")
    parser.add_argument("--leader-host", type=str, default="localhost",
                       help="P2P leader host (default: localhost)")
    parser.add_argument("--leader-port", type=int, default=8770,
                       help="P2P leader port (default: 8770)")
    parser.add_argument("--max-parallel", type=int, default=10,
                       help="Max parallel matches (default: 10)")
    parser.add_argument("--local-only", action="store_true",
                       help="Run locally without P2P distribution")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show matchups without playing")
    parser.add_argument("--leaderboard", action="store_true",
                       help="Show current leaderboard and exit")
    parser.add_argument("--status", action="store_true",
                       help="Check tournament status from P2P leader")

    args = parser.parse_args()

    tournament = P2PEloTournament(
        leader_host=args.leader_host,
        leader_port=args.leader_port,
        board_type=args.board,
        num_players=args.players,
        games_per_pairing=args.games,
    )

    if args.leaderboard:
        tournament.print_leaderboard()
        return

    if args.status:
        nodes = await tournament.discover_nodes()
        print("\n[Tournament] P2P Cluster Status")
        print(f"[Tournament] Found {len(nodes)} nodes")
        for node in nodes:
            gpu_info = f"GPU: {node.gpu_name}" if node.has_gpu else "CPU only"
            print(f"  - {node.node_id}: {node.host}:{node.port} ({gpu_info})")
        return

    if args.calibrate_ai_types:
        agents = list(AI_TYPE_CONFIGS.keys())
        print("\n[Tournament] AI Type Calibration Tournament")
        print(f"[Tournament] Agents: {agents}")

        if args.dry_run:
            matchups = tournament.generate_round_robin_matchups(agents)
            print(f"\n[Tournament] Would play {len(matchups)} matches:")
            for i, (a, b) in enumerate(matchups[:20]):
                print(f"  {i+1}. {a} vs {b}")
            if len(matchups) > 20:
                print(f"  ... and {len(matchups) - 20} more")
            return

        state = await tournament.run_tournament(
            agents=agents,
            max_parallel=args.max_parallel,
            use_distributed=not args.local_only,
        )
        tournament.print_leaderboard()

    elif args.models:
        # Discover models matching the board type and player count
        print(f"[Tournament] Discovering models for {args.board} {args.players}p...")
        models = discover_models(
            board_type=args.board,
            num_players=args.players,
        )

        if not models:
            print(f"[Tournament] No models found for {args.board} {args.players}p")
            print("[Tournament] Check that models exist in the models/ directory")
            return

        print(f"[Tournament] Found {len(models)} models:")
        for m in models[:10]:
            elo_str = f"(Elo: {m.elo:.0f})" if m.elo else ""
            print(f"  - {m.name} [{m.model_type}] {elo_str}")
        if len(models) > 10:
            print(f"  ... and {len(models) - 10} more")

        # Create model configs for tournament
        # Models are configured as neural net agents with checkpoint paths
        model_configs: dict[str, dict[str, Any]] = {}
        for m in models:
            model_configs[m.name] = {
                "ai_type": "descent",  # Use descent AI for NN models
                "model_path": m.path,
                "use_neural_net": True,
                "difficulty": 7,  # High difficulty for model evaluation
                "description": f"{m.model_type} model ({m.board_type})",
            }

        # Create tournament with model configs
        tournament = P2PEloTournament(
            board_type=args.board,
            num_players=args.players,
            games_per_pairing=args.games,
            leader_host=args.leader_host,
            leader_port=args.leader_port,
            elo_db_path=AI_SERVICE_ROOT / "data" / "elo" / f"model_elo_{args.board}_{args.players}p.db",
        )

        # Inject model configs into the global config dict for match execution
        # This allows run_match_locally and run_match_on_node to use model configs
        AI_TYPE_CONFIGS.update(model_configs)

        agents = list(model_configs.keys())
        matchups = tournament.generate_round_robin_matchups(agents)

        if args.dry_run:
            print(f"\n[Tournament] Would play {len(matchups)} matches:")
            for i, (a, b) in enumerate(matchups[:20]):
                print(f"  {i+1}. {a} vs {b}")
            if len(matchups) > 20:
                print(f"  ... and {len(matchups) - 20} more")
            return

        state = await tournament.run_tournament(
            agents=agents,
            max_parallel=args.max_parallel,
            use_distributed=not args.local_only,
        )

        # Print model-specific leaderboard
        print("\n" + "=" * 70)
        print("MODEL ELO LEADERBOARD")
        print("=" * 70)
        sorted_ratings = sorted(state.ratings.items(), key=lambda x: x[1], reverse=True)
        print(f"{'Rank':<6} {'Model':<40} {'Elo':<10}")
        print("-" * 70)
        for i, (model_name, rating) in enumerate(sorted_ratings, 1):
            print(f"{i:<6} {model_name[:40]:<40} {rating:>8.1f}")
        print("=" * 70)

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
