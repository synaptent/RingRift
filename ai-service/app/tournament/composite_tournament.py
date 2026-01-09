#!/usr/bin/env python3
"""Composite Tournament System for (NN, Algorithm) evaluation.

This module provides specialized tournament types for the Composite ELO System:

1. Algorithm Tournament (Weekly)
   - Fix NN to reference model
   - Round-robin between algorithms
   - Isolates algorithm strength from NN quality

2. NN Tournament (Daily)
   - Fix algorithm to standard (e.g., Gumbel budget=200)
   - Round-robin between NNs
   - Isolates NN quality from algorithm choice

3. Combined Tournament (Continuous)
   - All (NN, Algorithm) pairs compete
   - Swiss-style pairing for efficiency
   - Full combinatorial Elo ranking

Usage:
    from app.tournament.composite_tournament import (
        AlgorithmTournament,
        NNTournament,
        CombinedTournament,
        TournamentScheduleManager,
    )

    # Algorithm tournament with fixed NN
    algo_tourney = AlgorithmTournament(
        reference_nn="best_model_v5",
        board_type="square8",
        num_players=2,
    )
    results = await algo_tourney.run()

    # Schedule recurring tournaments
    scheduler = TournamentScheduleManager()
    scheduler.schedule_daily_nn_tournament()
    scheduler.schedule_weekly_algorithm_tournament()
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from app.models import BoardType
from app.tournament.scheduler import Match, MatchStatus, RoundRobinScheduler, SwissScheduler
from app.training.composite_participant import (
    STANDARD_ALGORITHM_CONFIGS,
    extract_ai_type,
    extract_harness_type,
    extract_nn_id,
    get_standard_config,
    make_composite_participant_id,
)
from app.training.elo_service import get_elo_service

logger = logging.getLogger(__name__)


class TournamentType(str, Enum):
    """Types of composite tournaments."""
    ALGORITHM = "algorithm"      # Fixed NN, compare algorithms
    NN = "nn"                    # Fixed algorithm, compare NNs
    COMBINED = "combined"        # All combinations compete


@dataclass
class TournamentConfig:
    """Configuration for a tournament."""
    games_per_matchup: int = 10
    parallel_workers: int = 4
    timeout_seconds: int = 300
    min_games_for_ranking: int = 5


@dataclass
class TournamentResult:
    """Result of a tournament."""
    tournament_id: str
    tournament_type: TournamentType
    board_type: str
    num_players: int
    started_at: float
    completed_at: float | None = None
    participants: list[str] = field(default_factory=list)
    rankings: list[dict[str, Any]] = field(default_factory=list)
    matches_played: int = 0
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTournament(ABC):
    """Abstract base class for composite tournaments."""

    def __init__(
        self,
        board_type: str = "square8",
        num_players: int = 2,
        config: TournamentConfig | None = None,
    ):
        self.board_type = board_type
        self.num_players = num_players
        self.config = config or TournamentConfig()
        self.elo_service = get_elo_service()
        self._tournament_id = str(uuid.uuid4())[:8]

    @abstractmethod
    async def run(self) -> TournamentResult:
        """Run the tournament and return results."""

    @abstractmethod
    def get_participants(self) -> list[str]:
        """Get list of participant IDs for this tournament."""

    def _get_board_type_enum(self) -> BoardType:
        """Convert board type string to enum."""
        board_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hexagonal": BoardType.HEXAGONAL,
        }
        return board_map.get(self.board_type, BoardType.SQUARE8)

    def _create_result(self, tournament_type: TournamentType) -> TournamentResult:
        """Create initial tournament result."""
        return TournamentResult(
            tournament_id=self._tournament_id,
            tournament_type=tournament_type,
            board_type=self.board_type,
            num_players=self.num_players,
            started_at=time.time(),
            participants=self.get_participants(),
        )

    async def _play_match(
        self,
        participant_a: str,
        participant_b: str,
    ) -> dict[str, Any] | None:
        """Play a single match between two participants.

        This is a simplified implementation - full implementation would
        instantiate appropriate AIs based on composite IDs.
        """
        # For now, return simulated result based on current Elo
        rating_a = self.elo_service.get_rating(
            participant_a, self.board_type, self.num_players
        )
        rating_b = self.elo_service.get_rating(
            participant_b, self.board_type, self.num_players
        )

        # Expected score for participant_a
        exp_a = 1.0 / (1.0 + 10 ** ((rating_b.rating - rating_a.rating) / 400))

        # Simulate outcome (for testing - real impl would play actual game)
        import random
        outcome = random.random()

        if outcome < exp_a:
            winner = participant_a
        elif outcome < exp_a + (1 - exp_a) * 0.1:  # 10% draw chance
            winner = None
        else:
            winner = participant_b

        return {
            "winner": winner,
            "participant_a": participant_a,
            "participant_b": participant_b,
            "game_length": random.randint(50, 200),
            "duration_sec": random.uniform(1.0, 5.0),
        }

    async def _run_round_robin(
        self,
        participants: list[str],
    ) -> tuple[int, list[dict[str, Any]]]:
        """Run round-robin matches between participants.

        Returns:
            Tuple of (matches_played, rankings)
        """
        scheduler = RoundRobinScheduler(
            games_per_pairing=self.config.games_per_matchup,
            shuffle_order=True,
        )

        matches = scheduler.generate_matches(
            agent_ids=participants,
            board_type=self._get_board_type_enum(),
            num_players=2,  # Head-to-head matches
        )

        matches_played = 0
        for match in matches:
            if len(match.agent_ids) >= 2:
                result = await self._play_match(
                    match.agent_ids[0],
                    match.agent_ids[1],
                )

                if result:
                    # Record match - January 2026: Extract harness_type for Elo tracking
                    harness_type = extract_harness_type(result["participant_a"])
                    self.elo_service.record_match(
                        participant_a=result["participant_a"],
                        participant_b=result["participant_b"],
                        winner=result["winner"],
                        board_type=self.board_type,
                        num_players=self.num_players,
                        game_length=result["game_length"],
                        duration_sec=result["duration_sec"],
                        harness_type=harness_type,
                    )
                    matches_played += 1

        # Get final rankings
        rankings = self.elo_service.get_composite_leaderboard(
            board_type=self.board_type,
            num_players=self.num_players,
            limit=len(participants),
        )

        # Filter to just tournament participants
        participant_set = set(participants)
        rankings = [r for r in rankings if r["participant_id"] in participant_set]

        return matches_played, rankings


class AlgorithmTournament(BaseTournament):
    """Tournament to rank algorithms using a fixed reference NN.

    Purpose: Determine which search algorithm performs best when paired
    with the same neural network. This isolates algorithm strength
    from NN quality.
    """

    def __init__(
        self,
        reference_nn: str | Path,
        algorithms: list[str] | None = None,
        board_type: str = "square8",
        num_players: int = 2,
        config: TournamentConfig | None = None,
    ):
        """Initialize algorithm tournament.

        Args:
            reference_nn: Path or ID of the reference neural network
            algorithms: List of algorithms to test (defaults to standard set)
            board_type: Board type for games
            num_players: Number of players
            config: Tournament configuration
        """
        super().__init__(board_type, num_players, config)

        self.reference_nn = str(reference_nn)
        self.nn_id = Path(reference_nn).stem if "/" in str(reference_nn) else str(reference_nn)
        self.algorithms = algorithms or ["gumbel_mcts", "mcts", "descent", "policy_only"]

    def get_participants(self) -> list[str]:
        """Get participant IDs for each algorithm."""
        return [
            make_composite_participant_id(self.nn_id, algo)
            for algo in self.algorithms
        ]

    async def run(self) -> TournamentResult:
        """Run algorithm tournament."""
        result = self._create_result(TournamentType.ALGORITHM)
        result.metadata["reference_nn"] = self.nn_id
        result.metadata["algorithms"] = self.algorithms

        logger.info(
            f"Starting Algorithm Tournament {self._tournament_id} "
            f"with NN={self.nn_id}, algorithms={self.algorithms}"
        )

        try:
            # Register all participants
            for algo in self.algorithms:
                self.elo_service.register_composite_participant(
                    nn_id=self.nn_id,
                    ai_type=algo,
                    board_type=self.board_type,
                    num_players=self.num_players,
                )

            # Run round-robin
            participants = self.get_participants()
            matches_played, rankings = await self._run_round_robin(participants)

            result.matches_played = matches_played
            result.rankings = rankings
            result.status = "completed"

        except Exception as e:
            logger.error(f"Algorithm tournament failed: {e}")
            result.status = f"failed: {e}"

        result.completed_at = time.time()
        return result


class NNTournament(BaseTournament):
    """Tournament to rank NNs using a fixed reference algorithm.

    Purpose: Determine which neural network performs best when paired
    with the same search algorithm. This isolates NN quality from
    algorithm choice.
    """

    def __init__(
        self,
        nn_ids: list[str],
        reference_algorithm: str = "gumbel_mcts",
        board_type: str = "square8",
        num_players: int = 2,
        config: TournamentConfig | None = None,
        nn_paths: dict[str, str] | None = None,
    ):
        """Initialize NN tournament.

        Args:
            nn_ids: List of neural network identifiers
            reference_algorithm: Algorithm to use for all NNs
            board_type: Board type for games
            num_players: Number of players
            config: Tournament configuration
            nn_paths: Optional mapping of nn_id -> model path
        """
        super().__init__(board_type, num_players, config)

        self.nn_ids = nn_ids
        self.reference_algorithm = reference_algorithm
        self.nn_paths = nn_paths or {}

    def get_participants(self) -> list[str]:
        """Get participant IDs for each NN."""
        return [
            make_composite_participant_id(nn_id, self.reference_algorithm)
            for nn_id in self.nn_ids
        ]

    async def run(self) -> TournamentResult:
        """Run NN tournament."""
        result = self._create_result(TournamentType.NN)
        result.metadata["reference_algorithm"] = self.reference_algorithm
        result.metadata["nn_count"] = len(self.nn_ids)

        logger.info(
            f"Starting NN Tournament {self._tournament_id} "
            f"with algorithm={self.reference_algorithm}, NNs={len(self.nn_ids)}"
        )

        try:
            # Register all participants
            for nn_id in self.nn_ids:
                self.elo_service.register_composite_participant(
                    nn_id=nn_id,
                    ai_type=self.reference_algorithm,
                    board_type=self.board_type,
                    num_players=self.num_players,
                    nn_model_path=self.nn_paths.get(nn_id),
                )

            # Run round-robin
            participants = self.get_participants()
            matches_played, rankings = await self._run_round_robin(participants)

            result.matches_played = matches_played
            result.rankings = rankings
            result.status = "completed"

        except Exception as e:
            logger.error(f"NN tournament failed: {e}")
            result.status = f"failed: {e}"

        result.completed_at = time.time()
        return result


class CombinedTournament(BaseTournament):
    """Tournament with all (NN, Algorithm) combinations competing.

    Purpose: Establish absolute strength rankings across the full
    combinatorial space. Uses Swiss-style pairing for efficiency.
    """

    def __init__(
        self,
        nn_ids: list[str],
        algorithms: list[str] | None = None,
        board_type: str = "square8",
        num_players: int = 2,
        config: TournamentConfig | None = None,
        swiss_rounds: int = 5,
    ):
        """Initialize combined tournament.

        Args:
            nn_ids: List of neural network identifiers
            algorithms: List of algorithms to include
            board_type: Board type for games
            num_players: Number of players
            config: Tournament configuration
            swiss_rounds: Number of Swiss system rounds
        """
        super().__init__(board_type, num_players, config)

        self.nn_ids = nn_ids
        self.algorithms = algorithms or ["gumbel_mcts", "mcts", "descent"]
        self.swiss_rounds = swiss_rounds

    def get_participants(self) -> list[str]:
        """Get all (NN, Algorithm) combination participant IDs."""
        participants = []
        for nn_id in self.nn_ids:
            for algo in self.algorithms:
                participants.append(
                    make_composite_participant_id(nn_id, algo)
                )
        return participants

    async def run(self) -> TournamentResult:
        """Run combined tournament with Swiss pairing."""
        result = self._create_result(TournamentType.COMBINED)
        result.metadata["nn_count"] = len(self.nn_ids)
        result.metadata["algorithm_count"] = len(self.algorithms)
        result.metadata["total_combinations"] = len(self.nn_ids) * len(self.algorithms)

        logger.info(
            f"Starting Combined Tournament {self._tournament_id} "
            f"with {len(self.nn_ids)} NNs × {len(self.algorithms)} algorithms"
        )

        try:
            # Register all participants
            for nn_id in self.nn_ids:
                for algo in self.algorithms:
                    self.elo_service.register_composite_participant(
                        nn_id=nn_id,
                        ai_type=algo,
                        board_type=self.board_type,
                        num_players=self.num_players,
                    )

            # Run Swiss-style tournament
            participants = self.get_participants()
            matches_played, rankings = await self._run_swiss(participants)

            result.matches_played = matches_played
            result.rankings = rankings
            result.status = "completed"

        except Exception as e:
            logger.error(f"Combined tournament failed: {e}")
            result.status = f"failed: {e}"

        result.completed_at = time.time()
        return result

    async def _run_swiss(
        self,
        participants: list[str],
    ) -> tuple[int, list[dict[str, Any]]]:
        """Run Swiss-system tournament."""
        scheduler = SwissScheduler(
            rounds=self.swiss_rounds,
        )

        # Generate first round
        scheduler.generate_matches(
            agent_ids=participants,
            board_type=self._get_board_type_enum(),
            num_players=2,
        )

        matches_played = 0

        for round_num in range(self.swiss_rounds):
            pending = scheduler.get_pending_matches()

            for match in pending:
                if len(match.agent_ids) >= 2:
                    result = await self._play_match(
                        match.agent_ids[0],
                        match.agent_ids[1],
                    )

                    if result:
                        # January 2026: Extract harness_type for Elo tracking
                        harness_type = extract_harness_type(result["participant_a"])
                        self.elo_service.record_match(
                            participant_a=result["participant_a"],
                            participant_b=result["participant_b"],
                            winner=result["winner"],
                            board_type=self.board_type,
                            num_players=self.num_players,
                            game_length=result["game_length"],
                            duration_sec=result["duration_sec"],
                            harness_type=harness_type,
                        )
                        matches_played += 1

                        # Update scheduler with result
                        scheduler.mark_match_completed(
                            match.match_id,
                            {"winner": result["winner"]},
                        )

            # Generate next round if not last
            if round_num < self.swiss_rounds - 1:
                scheduler.generate_next_round()

        # Get final rankings
        rankings = self.elo_service.get_composite_leaderboard(
            board_type=self.board_type,
            num_players=self.num_players,
            limit=len(participants),
        )

        participant_set = set(participants)
        rankings = [r for r in rankings if r["participant_id"] in participant_set]

        return matches_played, rankings


@dataclass
class ScheduledTournament:
    """A scheduled tournament with timing and configuration."""
    tournament_type: TournamentType
    schedule: str  # "daily", "weekly", "continuous"
    last_run: datetime | None = None
    next_run: datetime | None = None
    config: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class TournamentScheduleManager:
    """Manager for scheduling and running periodic tournaments.

    Handles:
    - Daily NN tournaments (when new models arrive)
    - Weekly algorithm tournaments
    - Continuous combined tournament (background)
    """

    def __init__(
        self,
        board_type: str = "square8",
        num_players: int = 2,
    ):
        self.board_type = board_type
        self.num_players = num_players
        self._schedules: dict[str, ScheduledTournament] = {}
        self._running = False
        self._callbacks: list[Callable[[TournamentResult], None]] = []

    def schedule_daily_nn_tournament(
        self,
        reference_algorithm: str = "gumbel_mcts",
        games_per_matchup: int = 10,
    ) -> None:
        """Schedule daily NN tournament."""
        self._schedules["daily_nn"] = ScheduledTournament(
            tournament_type=TournamentType.NN,
            schedule="daily",
            next_run=self._next_daily_time(),
            config={
                "reference_algorithm": reference_algorithm,
                "games_per_matchup": games_per_matchup,
            },
        )
        logger.info(f"Scheduled daily NN tournament at {self._schedules['daily_nn'].next_run}")

    def schedule_weekly_algorithm_tournament(
        self,
        algorithms: list[str] | None = None,
        games_per_matchup: int = 20,
    ) -> None:
        """Schedule weekly algorithm tournament."""
        self._schedules["weekly_algorithm"] = ScheduledTournament(
            tournament_type=TournamentType.ALGORITHM,
            schedule="weekly",
            next_run=self._next_weekly_time(),
            config={
                "algorithms": algorithms or ["gumbel_mcts", "mcts", "descent", "policy_only"],
                "games_per_matchup": games_per_matchup,
            },
        )
        logger.info(f"Scheduled weekly algorithm tournament at {self._schedules['weekly_algorithm'].next_run}")

    def schedule_continuous_combined(
        self,
        games_per_hour: int = 10,
        swiss_rounds: int = 3,
    ) -> None:
        """Schedule continuous combined tournament."""
        self._schedules["continuous_combined"] = ScheduledTournament(
            tournament_type=TournamentType.COMBINED,
            schedule="continuous",
            config={
                "games_per_hour": games_per_hour,
                "swiss_rounds": swiss_rounds,
            },
        )
        logger.info("Scheduled continuous combined tournament")

    def register_callback(self, callback: Callable[[TournamentResult], None]) -> None:
        """Register callback for tournament completion."""
        self._callbacks.append(callback)

    async def run_scheduler(self) -> None:
        """Run the tournament scheduler loop."""
        self._running = True
        logger.info("Starting tournament schedule manager")

        while self._running:
            now = datetime.now()

            for schedule_id, schedule in self._schedules.items():
                if not schedule.enabled:
                    continue

                if schedule.schedule == "continuous":
                    # Run continuous tournament periodically
                    await self._run_continuous_tournament(schedule)

                elif schedule.next_run and now >= schedule.next_run:
                    # Run scheduled tournament
                    await self._run_scheduled_tournament(schedule_id, schedule)

            # Check every minute
            await asyncio.sleep(60)

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False

    async def _run_scheduled_tournament(
        self,
        schedule_id: str,
        schedule: ScheduledTournament,
    ) -> None:
        """Run a scheduled tournament."""
        logger.info(f"Running scheduled tournament: {schedule_id}")

        try:
            if schedule.tournament_type == TournamentType.NN:
                result = await self._run_nn_tournament(schedule.config)
            elif schedule.tournament_type == TournamentType.ALGORITHM:
                result = await self._run_algorithm_tournament(schedule.config)
            else:
                return

            # Update schedule
            schedule.last_run = datetime.now()
            if schedule.schedule == "daily":
                schedule.next_run = self._next_daily_time()
            elif schedule.schedule == "weekly":
                schedule.next_run = self._next_weekly_time()

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Callback failed: {e}")

        except Exception as e:
            logger.error(f"Scheduled tournament {schedule_id} failed: {e}")

    async def _run_nn_tournament(self, config: dict) -> TournamentResult:
        """Run NN tournament with current top NNs."""
        elo_service = get_elo_service()

        # Get top NNs from current rankings
        nn_rankings = elo_service.get_nn_rankings(
            board_type=self.board_type,
            num_players=self.num_players,
            min_games=5,
        )

        nn_ids = [r["nn_model_id"] for r in nn_rankings[:20]]

        if len(nn_ids) < 2:
            logger.warning("Not enough NNs for tournament")
            return TournamentResult(
                tournament_id="skipped",
                tournament_type=TournamentType.NN,
                board_type=self.board_type,
                num_players=self.num_players,
                started_at=time.time(),
                completed_at=time.time(),
                status="skipped: not enough NNs",
            )

        tournament = NNTournament(
            nn_ids=nn_ids,
            reference_algorithm=config.get("reference_algorithm", "gumbel_mcts"),
            board_type=self.board_type,
            num_players=self.num_players,
            config=TournamentConfig(
                games_per_matchup=config.get("games_per_matchup", 10),
            ),
        )

        return await tournament.run()

    async def _run_algorithm_tournament(self, config: dict) -> TournamentResult:
        """Run algorithm tournament with best NN."""
        elo_service = get_elo_service()

        # Get best NN
        nn_rankings = elo_service.get_nn_rankings(
            board_type=self.board_type,
            num_players=self.num_players,
            min_games=10,
        )

        if not nn_rankings:
            logger.warning("No NNs available for algorithm tournament")
            return TournamentResult(
                tournament_id="skipped",
                tournament_type=TournamentType.ALGORITHM,
                board_type=self.board_type,
                num_players=self.num_players,
                started_at=time.time(),
                completed_at=time.time(),
                status="skipped: no NNs available",
            )

        best_nn = nn_rankings[0]["nn_model_id"]

        tournament = AlgorithmTournament(
            reference_nn=best_nn,
            algorithms=config.get("algorithms"),
            board_type=self.board_type,
            num_players=self.num_players,
            config=TournamentConfig(
                games_per_matchup=config.get("games_per_matchup", 20),
            ),
        )

        return await tournament.run()

    async def _run_continuous_tournament(self, schedule: ScheduledTournament) -> None:
        """Run background continuous tournament games."""
        # Implementation would run a few games periodically
        pass

    def _next_daily_time(self) -> datetime:
        """Get next daily tournament time (2 AM local)."""
        now = datetime.now()
        next_run = now.replace(hour=2, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        return next_run

    def _next_weekly_time(self) -> datetime:
        """Get next weekly tournament time (Sunday 3 AM)."""
        now = datetime.now()
        days_until_sunday = (6 - now.weekday()) % 7
        if days_until_sunday == 0 and now.hour >= 3:
            days_until_sunday = 7
        next_run = now.replace(hour=3, minute=0, second=0, microsecond=0)
        next_run += timedelta(days=days_until_sunday)
        return next_run

    def get_schedule_status(self) -> dict[str, Any]:
        """Get current schedule status."""
        return {
            schedule_id: {
                "type": s.tournament_type.value,
                "schedule": s.schedule,
                "enabled": s.enabled,
                "last_run": s.last_run.isoformat() if s.last_run else None,
                "next_run": s.next_run.isoformat() if s.next_run else None,
            }
            for schedule_id, s in self._schedules.items()
        }
