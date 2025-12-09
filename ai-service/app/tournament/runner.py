"""Tournament match execution and result aggregation."""
from __future__ import annotations

import asyncio
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.models import BoardType, GameStatus

from .agents import AIAgent, AIAgentRegistry
from .elo import EloCalculator
from .scheduler import Match, MatchStatus, TournamentScheduler

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a single match."""

    match_id: str
    agent_ids: List[str]
    rankings: List[str]  # Ordered by finish position (1st, 2nd, ...)
    winner: Optional[str]  # Agent ID of winner, None for draw/timeout
    game_length: int
    termination_reason: str
    duration_seconds: float
    metadata: Dict = field(default_factory=dict)

    @property
    def is_draw(self) -> bool:
        return self.winner is None and "draw" in self.termination_reason.lower()

    def to_dict(self) -> Dict:
        return {
            "match_id": self.match_id,
            "agent_ids": self.agent_ids,
            "rankings": self.rankings,
            "winner": self.winner,
            "game_length": self.game_length,
            "termination_reason": self.termination_reason,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


@dataclass
class TournamentResults:
    """Aggregated results for a tournament."""

    tournament_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    match_results: List[MatchResult] = field(default_factory=list)
    final_ratings: Dict[str, float] = field(default_factory=dict)
    agent_stats: Dict[str, Dict] = field(default_factory=dict)

    def add_result(self, result: MatchResult) -> None:
        self.match_results.append(result)

    def compute_stats(self) -> None:
        """Compute per-agent statistics."""
        self.agent_stats = {}

        for result in self.match_results:
            for idx, agent_id in enumerate(result.rankings):
                if agent_id not in self.agent_stats:
                    self.agent_stats[agent_id] = {
                        "games_played": 0,
                        "wins": 0,
                        "losses": 0,
                        "draws": 0,
                        "total_game_length": 0,
                        "positions": [],
                    }

                stats = self.agent_stats[agent_id]
                stats["games_played"] += 1
                stats["total_game_length"] += result.game_length
                stats["positions"].append(idx + 1)

                num_players = len(result.rankings)
                if idx == 0:
                    stats["wins"] += 1
                elif idx == num_players - 1:
                    stats["losses"] += 1
                elif result.is_draw:
                    stats["draws"] += 1

        # Compute derived stats
        for agent_id, stats in self.agent_stats.items():
            if stats["games_played"] > 0:
                stats["win_rate"] = stats["wins"] / stats["games_played"]
                stats["avg_game_length"] = (
                    stats["total_game_length"] / stats["games_played"]
                )
                stats["avg_position"] = sum(stats["positions"]) / len(stats["positions"])
            else:
                stats["win_rate"] = 0.0
                stats["avg_game_length"] = 0.0
                stats["avg_position"] = 0.0

    def to_dict(self) -> Dict:
        return {
            "tournament_id": self.tournament_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "total_matches": len(self.match_results),
            "final_ratings": self.final_ratings,
            "agent_stats": self.agent_stats,
            "match_results": [r.to_dict() for r in self.match_results],
        }


class TournamentRunner:
    """Executes tournament matches and manages Elo ratings.

    Supports both local execution and distributed execution via workers.
    """

    def __init__(
        self,
        agent_registry: AIAgentRegistry,
        scheduler: TournamentScheduler,
        elo_calculator: Optional[EloCalculator] = None,
        max_workers: int = 4,
        max_moves: int = 500,
        seed: Optional[int] = None,
    ):
        """Initialize tournament runner.

        Args:
            agent_registry: Registry of AI agents.
            scheduler: Match scheduler.
            elo_calculator: Elo rating calculator (created if None).
            max_workers: Maximum parallel workers for local execution.
            max_moves: Maximum moves per game before timeout.
            seed: Random seed for reproducibility.
        """
        self.agent_registry = agent_registry
        self.scheduler = scheduler
        self.elo_calculator = elo_calculator or EloCalculator()
        self.max_workers = max_workers
        self.max_moves = max_moves
        self.seed = seed
        self._rng = random.Random(seed)

        self.results: Optional[TournamentResults] = None
        self._match_executor: Optional[Callable] = None

    def set_match_executor(
        self,
        executor: Callable[[Match, Dict[str, AIAgent]], MatchResult],
    ) -> None:
        """Set custom match executor for distributed execution.

        Args:
            executor: Function that takes (match, agents) and returns MatchResult.
        """
        self._match_executor = executor

    def run_tournament(
        self,
        agent_ids: List[str],
        board_type: BoardType,
        num_players: int = 2,
        games_per_pairing: int = 2,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> TournamentResults:
        """Run a complete tournament.

        Args:
            agent_ids: List of agent IDs to compete.
            board_type: Board type for all matches.
            num_players: Number of players per match.
            games_per_pairing: Number of games per agent pairing.
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            Tournament results with final ratings.
        """
        from uuid import uuid4

        # Validate agents exist
        agents = {}
        for agent_id in agent_ids:
            agent = self.agent_registry.get(agent_id)
            if agent is None:
                raise ValueError(f"Agent not found: {agent_id}")
            agents[agent_id] = agent

        # Generate matches
        self.scheduler.reset() if hasattr(self.scheduler, "reset") else None
        matches = self.scheduler.generate_matches(
            agent_ids=agent_ids,
            board_type=board_type,
            num_players=num_players,
            games_per_pairing=games_per_pairing,
        )

        logger.info(
            f"Generated {len(matches)} matches for {len(agent_ids)} agents"
        )

        # Initialize results
        self.results = TournamentResults(
            tournament_id=str(uuid4()),
            started_at=datetime.now(),
        )

        # Execute matches
        total_matches = len(matches)
        completed = 0

        if self._match_executor:
            # Use custom executor (for distributed execution)
            for match in matches:
                try:
                    result = self._match_executor(match, agents)
                    self._process_result(match, result)
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total_matches)
                except Exception as e:
                    logger.error(f"Match {match.match_id} failed: {e}")
                    self.scheduler.mark_match_failed(match.match_id, str(e))
        else:
            # Local parallel execution
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._execute_match_local, match, agents
                    ): match
                    for match in matches
                }

                for future in as_completed(futures):
                    match = futures[future]
                    try:
                        result = future.result()
                        self._process_result(match, result)
                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total_matches)
                    except Exception as e:
                        logger.error(f"Match {match.match_id} failed: {e}")
                        self.scheduler.mark_match_failed(match.match_id, str(e))

        # Finalize
        self.results.completed_at = datetime.now()
        self.results.compute_stats()

        # Store final ratings
        for agent_id in agent_ids:
            rating = self.elo_calculator.get_rating(agent_id)
            self.results.final_ratings[agent_id] = rating.rating

        return self.results

    def _process_result(self, match: Match, result: MatchResult) -> None:
        """Process a match result - update scheduler, Elo, and results."""
        # Update scheduler
        self.scheduler.mark_match_completed(
            match.match_id,
            {
                "rankings": result.rankings,
                "winner": result.winner,
                "game_length": result.game_length,
            },
        )

        # Update Elo ratings
        if len(result.rankings) == 2:
            # 2-player: use standard update
            if result.winner == result.agent_ids[0]:
                score = 1.0
            elif result.winner == result.agent_ids[1]:
                score = 0.0
            else:
                score = 0.5
            self.elo_calculator.update_ratings(
                result.agent_ids[0], result.agent_ids[1], score
            )
        else:
            # Multiplayer: use ranking-based update
            self.elo_calculator.update_multiplayer_ratings(result.rankings)

        # Store result
        self.results.add_result(result)

    def _execute_match_local(
        self,
        match: Match,
        agents: Dict[str, AIAgent],
    ) -> MatchResult:
        """Execute a single match locally.

        This is the default local executor. Override with set_match_executor
        for distributed execution.
        """
        import time

        from app.ai.search import MinimaxSearch
        from app.board_manager import BoardManager
        from app.game_engine import GameEngine
        from app.models import GameState

        start_time = time.time()

        # Mark match as started
        self.scheduler.mark_match_started(match.match_id)

        # Initialize game
        state = BoardManager.create_initial_state(
            board_type=match.board_type,
            num_players=match.num_players,
        )

        # Create AI instances for each agent
        ai_instances = []
        for agent_id in match.agent_ids:
            agent = agents[agent_id]
            ai = MinimaxSearch(
                weights=agent.weights,
                max_depth=agent.search_depth,
            )
            ai_instances.append(ai)

        # Play the game
        move_count = 0
        while state.game_status == GameStatus.ACTIVE and move_count < self.max_moves:
            current_player = state.current_player
            ai = ai_instances[current_player]

            # Get AI move
            legal_moves = GameEngine.get_legal_moves(state)
            if not legal_moves:
                break

            move = ai.get_best_move(state, legal_moves)
            state = GameEngine.apply_move(state, move)
            move_count += 1

        # Determine rankings based on elimination order or ring counts
        rankings = self._compute_rankings(state, match.agent_ids)
        winner = rankings[0] if state.winner is not None else None

        # Determine termination reason
        if state.game_status == GameStatus.COMPLETED:
            termination_reason = "completed"
        elif move_count >= self.max_moves:
            termination_reason = "max_moves"
        else:
            termination_reason = "no_moves"

        duration = time.time() - start_time

        return MatchResult(
            match_id=match.match_id,
            agent_ids=match.agent_ids,
            rankings=rankings,
            winner=winner,
            game_length=move_count,
            termination_reason=termination_reason,
            duration_seconds=duration,
        )

    def _compute_rankings(
        self,
        state: Any,  # GameState
        agent_ids: List[str],
    ) -> List[str]:
        """Compute player rankings from final game state."""
        num_players = len(agent_ids)

        # If there's a winner, they're first
        if state.winner is not None:
            rankings = [agent_ids[state.winner]]
            remaining = [
                agent_ids[i] for i in range(num_players) if i != state.winner
            ]
            # Sort remaining by elimination order (last eliminated = higher rank)
            # For now, just append them in reverse order
            rankings.extend(reversed(remaining))
            return rankings

        # For multiplayer without clear winner, rank by ring count
        ring_counts = []
        for player_idx in range(num_players):
            count = sum(
                1
                for ring in state.rings
                if ring.player == player_idx and ring.position is not None
            )
            ring_counts.append((player_idx, count))

        # Sort by ring count descending
        ring_counts.sort(key=lambda x: x[1], reverse=True)
        return [agent_ids[player_idx] for player_idx, _ in ring_counts]

    def get_leaderboard(self) -> List[Tuple[str, float, Dict]]:
        """Get current leaderboard with ratings and stats.

        Returns:
            List of (agent_id, rating, stats) tuples sorted by rating.
        """
        leaderboard = []
        for rating in self.elo_calculator.get_leaderboard():
            stats = (
                self.results.agent_stats.get(rating.agent_id, {})
                if self.results
                else {}
            )
            leaderboard.append((rating.agent_id, rating.rating, stats))
        return leaderboard

    def save_results(self, path: Path) -> None:
        """Save tournament results to JSON file."""
        if not self.results:
            raise ValueError("No results to save")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results.to_dict(), f, indent=2)

    def load_results(self, path: Path) -> TournamentResults:
        """Load tournament results from JSON file."""
        with open(path) as f:
            data = json.load(f)

        results = TournamentResults(
            tournament_id=data["tournament_id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data["completed_at"]
                else None
            ),
        )
        results.final_ratings = data["final_ratings"]
        results.agent_stats = data["agent_stats"]

        for mr_data in data.get("match_results", []):
            results.match_results.append(
                MatchResult(
                    match_id=mr_data["match_id"],
                    agent_ids=mr_data["agent_ids"],
                    rankings=mr_data["rankings"],
                    winner=mr_data["winner"],
                    game_length=mr_data["game_length"],
                    termination_reason=mr_data["termination_reason"],
                    duration_seconds=mr_data["duration_seconds"],
                    metadata=mr_data.get("metadata", {}),
                )
            )

        self.results = results
        return results
