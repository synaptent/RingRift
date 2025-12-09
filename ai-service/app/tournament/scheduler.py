"""Tournament scheduling for AI agent matchups."""
from __future__ import annotations

import itertools
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from app.models import BoardType


class MatchStatus(str, Enum):
    """Status of a scheduled match."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Match:
    """Represents a scheduled match between agents."""

    match_id: str
    agent_ids: List[str]  # For multiplayer, can have 2-4 agents
    board_type: BoardType
    round_number: int = 0
    status: MatchStatus = MatchStatus.PENDING
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict] = None  # Rankings, winner, etc.
    worker_id: Optional[str] = None  # Which worker executed this match
    metadata: Dict = field(default_factory=dict)

    @property
    def num_players(self) -> int:
        return len(self.agent_ids)

    @property
    def is_multiplayer(self) -> bool:
        return len(self.agent_ids) > 2

    def to_dict(self) -> Dict:
        return {
            "match_id": self.match_id,
            "agent_ids": self.agent_ids,
            "board_type": self.board_type.value,
            "round_number": self.round_number,
            "status": self.status.value,
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "worker_id": self.worker_id,
            "metadata": self.metadata,
        }


class TournamentScheduler(ABC):
    """Abstract base class for tournament schedulers."""

    @abstractmethod
    def generate_matches(
        self,
        agent_ids: List[str],
        board_type: BoardType,
        **kwargs,
    ) -> List[Match]:
        """Generate all matches for the tournament."""
        pass

    @abstractmethod
    def get_pending_matches(self) -> List[Match]:
        """Get matches that haven't been played yet."""
        pass

    @abstractmethod
    def mark_match_started(self, match_id: str, worker_id: Optional[str] = None) -> None:
        """Mark a match as in progress."""
        pass

    @abstractmethod
    def mark_match_completed(self, match_id: str, result: Dict) -> None:
        """Mark a match as completed with results."""
        pass


class RoundRobinScheduler(TournamentScheduler):
    """Round-robin tournament scheduler.

    For 2-player games: Each agent plays every other agent.
    For multiplayer: Generates all combinations of N agents from the pool.
    """

    def __init__(
        self,
        games_per_pairing: int = 2,
        shuffle_order: bool = True,
        seed: Optional[int] = None,
    ):
        """Initialize round-robin scheduler.

        Args:
            games_per_pairing: Number of games per agent pairing (2 allows each
                              agent to play both sides/positions).
            shuffle_order: Whether to shuffle match order.
            seed: Random seed for reproducibility.
        """
        self.games_per_pairing = games_per_pairing
        self.shuffle_order = shuffle_order
        self.seed = seed
        self._matches: Dict[str, Match] = {}
        self._rng = random.Random(seed)

    def generate_matches(
        self,
        agent_ids: List[str],
        board_type: BoardType,
        num_players: int = 2,
        **kwargs,
    ) -> List[Match]:
        """Generate round-robin matches.

        Args:
            agent_ids: List of agent IDs to include.
            board_type: Board type for all matches.
            num_players: Number of players per match (2, 3, or 4).

        Returns:
            List of generated matches.
        """
        if len(agent_ids) < num_players:
            raise ValueError(
                f"Need at least {num_players} agents for {num_players}-player matches"
            )

        matches = []
        round_num = 0
        now = datetime.now()

        # Generate all combinations of agents for this player count
        for combination in itertools.combinations(agent_ids, num_players):
            agent_list = list(combination)

            # Generate multiple games per pairing
            for game_num in range(self.games_per_pairing):
                # Rotate player order for fairness
                if game_num > 0:
                    agent_list = agent_list[1:] + agent_list[:1]

                match = Match(
                    match_id=str(uuid4()),
                    agent_ids=agent_list.copy(),
                    board_type=board_type,
                    round_number=round_num,
                    scheduled_at=now,
                    metadata={"game_num": game_num},
                )
                matches.append(match)

            round_num += 1

        # Shuffle match order if requested
        if self.shuffle_order:
            self._rng.shuffle(matches)

        # Store matches
        for match in matches:
            self._matches[match.match_id] = match

        return matches

    def get_pending_matches(self) -> List[Match]:
        """Get all pending matches."""
        return [m for m in self._matches.values() if m.status == MatchStatus.PENDING]

    def get_matches_by_status(self, status: MatchStatus) -> List[Match]:
        """Get matches by status."""
        return [m for m in self._matches.values() if m.status == status]

    def get_match(self, match_id: str) -> Optional[Match]:
        """Get match by ID."""
        return self._matches.get(match_id)

    def mark_match_started(
        self,
        match_id: str,
        worker_id: Optional[str] = None,
    ) -> None:
        """Mark a match as in progress."""
        match = self._matches.get(match_id)
        if match:
            match.status = MatchStatus.IN_PROGRESS
            match.started_at = datetime.now()
            match.worker_id = worker_id

    def mark_match_completed(self, match_id: str, result: Dict) -> None:
        """Mark a match as completed with results."""
        match = self._matches.get(match_id)
        if match:
            match.status = MatchStatus.COMPLETED
            match.completed_at = datetime.now()
            match.result = result

    def mark_match_failed(self, match_id: str, error: str) -> None:
        """Mark a match as failed."""
        match = self._matches.get(match_id)
        if match:
            match.status = MatchStatus.FAILED
            match.completed_at = datetime.now()
            match.result = {"error": error}

    def get_all_matches(self) -> List[Match]:
        """Get all matches."""
        return list(self._matches.values())

    def get_stats(self) -> Dict:
        """Get tournament progress statistics."""
        total = len(self._matches)
        by_status = {status: 0 for status in MatchStatus}
        for match in self._matches.values():
            by_status[match.status] += 1

        return {
            "total_matches": total,
            "pending": by_status[MatchStatus.PENDING],
            "in_progress": by_status[MatchStatus.IN_PROGRESS],
            "completed": by_status[MatchStatus.COMPLETED],
            "failed": by_status[MatchStatus.FAILED],
            "cancelled": by_status[MatchStatus.CANCELLED],
            "completion_pct": (
                by_status[MatchStatus.COMPLETED] / total * 100 if total > 0 else 0
            ),
        }

    def reset(self) -> None:
        """Clear all matches."""
        self._matches.clear()


class SwissScheduler(TournamentScheduler):
    """Swiss-system tournament scheduler.

    Pairs agents with similar scores, reducing total games needed
    while still producing reliable rankings.
    """

    def __init__(
        self,
        rounds: int = 5,
        seed: Optional[int] = None,
    ):
        """Initialize Swiss scheduler.

        Args:
            rounds: Number of Swiss rounds to play.
            seed: Random seed for reproducibility.
        """
        self.rounds = rounds
        self.seed = seed
        self._matches: Dict[str, Match] = {}
        self._scores: Dict[str, float] = {}
        self._pairings_history: Dict[Tuple[str, str], int] = {}
        self._current_round = 0
        self._rng = random.Random(seed)

    def generate_matches(
        self,
        agent_ids: List[str],
        board_type: BoardType,
        num_players: int = 2,
        **kwargs,
    ) -> List[Match]:
        """Generate first round of Swiss matches.

        For subsequent rounds, call generate_next_round() after
        completing the current round.
        """
        if num_players != 2:
            raise ValueError("Swiss system currently only supports 2-player matches")

        # Initialize scores
        for agent_id in agent_ids:
            self._scores[agent_id] = 0.0

        return self._generate_round(agent_ids, board_type)

    def _generate_round(
        self,
        agent_ids: List[str],
        board_type: BoardType,
    ) -> List[Match]:
        """Generate matches for one Swiss round."""
        # Sort agents by score (descending), with random tiebreaker
        sorted_agents = sorted(
            agent_ids,
            key=lambda a: (self._scores.get(a, 0), self._rng.random()),
            reverse=True,
        )

        matches = []
        paired = set()
        now = datetime.now()

        # Greedy pairing: pair top unpaired with next best unpaired
        for agent in sorted_agents:
            if agent in paired:
                continue

            # Find best opponent not yet paired and not played too often
            for opponent in sorted_agents:
                if opponent == agent or opponent in paired:
                    continue

                pair_key = tuple(sorted([agent, opponent]))
                times_played = self._pairings_history.get(pair_key, 0)

                # Avoid re-pairing if possible
                if times_played < 2:  # Allow max 2 games between same pair
                    match = Match(
                        match_id=str(uuid4()),
                        agent_ids=[agent, opponent],
                        board_type=board_type,
                        round_number=self._current_round,
                        scheduled_at=now,
                    )
                    matches.append(match)
                    self._matches[match.match_id] = match
                    paired.add(agent)
                    paired.add(opponent)
                    self._pairings_history[pair_key] = times_played + 1
                    break

        self._current_round += 1
        return matches

    def generate_next_round(
        self,
        agent_ids: List[str],
        board_type: BoardType,
    ) -> List[Match]:
        """Generate next round based on current scores."""
        if self._current_round >= self.rounds:
            return []
        return self._generate_round(agent_ids, board_type)

    def update_scores(self, match_id: str, result: Dict) -> None:
        """Update scores after a match."""
        match = self._matches.get(match_id)
        if not match or len(match.agent_ids) != 2:
            return

        agent_a, agent_b = match.agent_ids
        winner = result.get("winner")

        if winner == 0:  # Agent A wins
            self._scores[agent_a] = self._scores.get(agent_a, 0) + 1.0
        elif winner == 1:  # Agent B wins
            self._scores[agent_b] = self._scores.get(agent_b, 0) + 1.0
        else:  # Draw
            self._scores[agent_a] = self._scores.get(agent_a, 0) + 0.5
            self._scores[agent_b] = self._scores.get(agent_b, 0) + 0.5

    def get_pending_matches(self) -> List[Match]:
        return [m for m in self._matches.values() if m.status == MatchStatus.PENDING]

    def mark_match_started(
        self,
        match_id: str,
        worker_id: Optional[str] = None,
    ) -> None:
        match = self._matches.get(match_id)
        if match:
            match.status = MatchStatus.IN_PROGRESS
            match.started_at = datetime.now()
            match.worker_id = worker_id

    def mark_match_completed(self, match_id: str, result: Dict) -> None:
        match = self._matches.get(match_id)
        if match:
            match.status = MatchStatus.COMPLETED
            match.completed_at = datetime.now()
            match.result = result
            self.update_scores(match_id, result)

    def get_standings(self) -> List[Tuple[str, float]]:
        """Get current standings sorted by score."""
        return sorted(
            self._scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

    def is_tournament_complete(self) -> bool:
        """Check if all rounds are complete."""
        return self._current_round >= self.rounds and not self.get_pending_matches()
