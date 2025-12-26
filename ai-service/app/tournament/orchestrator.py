"""Unified Tournament Orchestrator for RingRift.

This module provides a high-level interface for running AI tournaments,
integrating with the tournament framework, Elo database, and metrics system.

Usage:
    from app.tournament.orchestrator import TournamentOrchestrator

    # Create orchestrator
    orchestrator = TournamentOrchestrator(
        board_type="square8",
        num_players=2,
    )

    # Run a quick evaluation tournament
    results = orchestrator.run_evaluation(
        candidate_model="model_v2",
        baseline_models=["model_v1", "random", "heuristic"],
        games_per_pairing=20,
    )

    # Run a full round-robin tournament
    results = orchestrator.run_round_robin(
        agents=["random", "heuristic", "mcts_100", "mcts_500"],
        games_per_pairing=50,
    )

    # Run a shadow evaluation (quick check during training)
    passed = orchestrator.run_shadow_eval(
        candidate="new_model",
        games=15,
        elo_threshold=25,
    )
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Event bus integration for training feedback (December 2025)
try:
    from app.coordination.event_router import (
        emit_elo_updated,
    )
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False
    emit_elo_updated = None

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of a model evaluation."""
    candidate_id: str
    baseline_id: str
    elo_delta: float
    win_rate: float
    games_played: int
    passed: bool
    confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TournamentSummary:
    """Summary of a tournament run."""
    tournament_id: str
    board_type: str
    num_players: int
    started_at: datetime
    completed_at: datetime
    total_games: int
    duration_seconds: float
    final_ratings: dict[str, float]
    win_rates: dict[str, float]
    agent_stats: dict[str, dict]
    evaluation_results: list[EvaluationResult] = field(default_factory=list)


class TournamentOrchestrator:
    """High-level tournament orchestration.

    This class provides simplified APIs for common tournament scenarios,
    handling all the setup and integration automatically.
    """

    def __init__(
        self,
        board_type: str = "square8",
        num_players: int = 2,
        *,
        max_workers: int = 4,
        persist_to_elo_db: bool = True,
        record_metrics: bool = True,
    ):
        """Initialize the tournament orchestrator.

        Args:
            board_type: Board type for tournaments
            num_players: Number of players per game
            max_workers: Maximum parallel workers
            persist_to_elo_db: Persist results to unified Elo database
            record_metrics: Record Prometheus metrics
        """
        self.board_type = board_type
        self.num_players = num_players
        self.max_workers = max_workers
        self.persist_to_elo_db = persist_to_elo_db
        self.record_metrics = record_metrics

        # Lazy-loaded dependencies
        self._runner = None
        self._elo_db = None
        self._elo_service = None

    @property
    def runner(self):
        """Lazy-load tournament runner."""
        if self._runner is None:
            from app.tournament import create_tournament_runner
            self._runner = create_tournament_runner(
                scheduler_type="round_robin",
                max_workers=self.max_workers,
                persist_to_unified_elo=self.persist_to_elo_db,
            )
        return self._runner

    @property
    def elo_db(self):
        """Lazy-load Elo database."""
        if self._elo_db is None:
            from app.tournament import get_elo_database
            self._elo_db = get_elo_database()
        return self._elo_db

    @property
    def elo_service(self):
        """Lazy-load EloService (preferred)."""
        if self._elo_service is None:
            try:
                from app.training.elo_service import get_elo_service
                self._elo_service = get_elo_service()
            except Exception as e:
                logger.debug(f"EloService unavailable: {e}")
                self._elo_service = None
        return self._elo_service

    def run_round_robin(
        self,
        agents: list[str],
        games_per_pairing: int = 20,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> TournamentSummary:
        """Run a full round-robin tournament.

        Args:
            agents: List of agent IDs to compete
            games_per_pairing: Games per agent pairing
            progress_callback: Optional progress callback

        Returns:
            TournamentSummary with results
        """
        from app.models import BoardType
        from app.tournament import TournamentResults

        tournament_id = f"roundrobin_{uuid.uuid4().hex[:8]}"
        started_at = datetime.now()

        logger.info(
            f"Starting round-robin tournament {tournament_id}: "
            f"{len(agents)} agents, {games_per_pairing} games/pairing"
        )

        # Run tournament
        board_type_enum = BoardType(self.board_type)
        results: TournamentResults = self.runner.run_tournament(
            agent_ids=agents,
            board_type=board_type_enum,
            num_players=self.num_players,
            games_per_pairing=games_per_pairing,
            progress_callback=progress_callback,
        )

        completed_at = datetime.now()
        duration = (completed_at - started_at).total_seconds()

        # Compute win rates
        results.compute_stats()
        win_rates = {
            agent_id: stats.get("win_rate", 0.0)
            for agent_id, stats in results.agent_stats.items()
        }

        # Record metrics
        if self.record_metrics:
            self._record_tournament_metrics(
                tournament_type="round_robin",
                games=len(results.match_results),
                duration=duration,
            )

        summary = TournamentSummary(
            tournament_id=tournament_id,
            board_type=self.board_type,
            num_players=self.num_players,
            started_at=started_at,
            completed_at=completed_at,
            total_games=len(results.match_results),
            duration_seconds=duration,
            final_ratings=results.final_ratings,
            win_rates=win_rates,
            agent_stats=results.agent_stats,
        )

        logger.info(
            f"Tournament {tournament_id} complete: "
            f"{summary.total_games} games in {duration:.1f}s"
        )

        return summary

    def run_evaluation(
        self,
        candidate_model: str,
        baseline_models: list[str],
        games_per_pairing: int = 20,
        *,
        elo_threshold: float = 25.0,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[TournamentSummary, list[EvaluationResult]]:
        """Run an evaluation tournament for a candidate model.

        Args:
            candidate_model: Candidate model to evaluate
            baseline_models: Baseline models to compare against
            games_per_pairing: Games per pairing
            elo_threshold: Elo threshold to pass evaluation
            progress_callback: Progress callback

        Returns:
            Tuple of (TournamentSummary, list of EvaluationResult)
        """
        all_agents = [candidate_model, *baseline_models]

        # Run round-robin
        summary = self.run_round_robin(
            agents=all_agents,
            games_per_pairing=games_per_pairing,
            progress_callback=progress_callback,
        )

        # Compute evaluation results
        evaluation_results = []
        candidate_elo = summary.final_ratings.get(candidate_model, 1500)
        candidate_win_rate = summary.win_rates.get(candidate_model, 0.5)

        for baseline in baseline_models:
            baseline_elo = summary.final_ratings.get(baseline, 1500)
            elo_delta = candidate_elo - baseline_elo

            # Get head-to-head stats
            for _result in summary.agent_stats.get(candidate_model, {}).get("match_results", []):
                # This is simplified - actual h2h would need match filtering
                pass

            eval_result = EvaluationResult(
                candidate_id=candidate_model,
                baseline_id=baseline,
                elo_delta=elo_delta,
                win_rate=candidate_win_rate,
                games_played=summary.total_games,
                passed=elo_delta >= elo_threshold,
                confidence=self._compute_confidence(summary.total_games, elo_delta),
            )
            evaluation_results.append(eval_result)

        summary.evaluation_results = evaluation_results

        # Record evaluation metrics
        if self.record_metrics:
            for eval_result in evaluation_results:
                self._record_evaluation_metrics(eval_result)

        # Emit evaluation events for training feedback (December 2025)
        self._emit_evaluation_events(candidate_model, evaluation_results, summary)

        return summary, evaluation_results

    def _emit_evaluation_events(
        self,
        candidate_id: str,
        evaluation_results: list[EvaluationResult],
        summary: TournamentSummary,
    ) -> None:
        """Emit events after evaluation for training feedback loop (December 2025).

        Emits ELO_UPDATED for each evaluation result, which triggers:
        - PromotionController to check for auto-promotion
        - CurriculumFeedback to adjust training priorities
        - RollbackManager to detect regressions
        """
        if not HAS_EVENT_BUS or emit_elo_updated is None:
            return

        try:
            for eval_result in evaluation_results:
                # Emit Elo update for the candidate vs each baseline
                try:
                    asyncio.get_running_loop().create_task(
                        emit_elo_updated(
                            model_id=eval_result.candidate_id,
                            new_elo=summary.final_ratings.get(eval_result.candidate_id, 1500),
                            elo_delta=eval_result.elo_delta,
                            games_played=eval_result.games_played,
                            opponent_id=eval_result.baseline_id,
                            source="tournament_orchestrator",
                        )
                    )
                except RuntimeError:
                    # No event loop running
                    pass

            logger.debug(
                f"[TournamentOrchestrator] Emitted {len(evaluation_results)} "
                f"evaluation events for {candidate_id}"
            )
        except Exception as e:
            logger.warning(f"[TournamentOrchestrator] Failed to emit events: {e}")

    def run_shadow_eval(
        self,
        candidate: str,
        games: int = 15,
        *,
        baselines: list[str] | None = None,
        elo_threshold: float = 25.0,
    ) -> bool:
        """Run a quick shadow evaluation.

        This is designed for quick checks during training iterations.

        Args:
            candidate: Candidate model ID
            games: Number of games per baseline
            baselines: Baseline models (defaults to ["heuristic", "mcts_100"])
            elo_threshold: Elo threshold to pass

        Returns:
            True if candidate passes threshold against all baselines
        """
        if baselines is None:
            baselines = ["heuristic", "mcts_100"]

        logger.info(f"Running shadow evaluation for {candidate}")

        _, eval_results = self.run_evaluation(
            candidate_model=candidate,
            baseline_models=baselines,
            games_per_pairing=games,
            elo_threshold=elo_threshold,
        )

        passed = all(r.passed for r in eval_results)

        logger.info(
            f"Shadow eval {candidate}: {'PASSED' if passed else 'FAILED'} "
            f"(threshold: {elo_threshold} Elo)"
        )

        return passed

    def get_current_elo(self, agent_id: str) -> float | None:
        """Get current Elo rating for an agent from the database.

        Args:
            agent_id: Agent identifier

        Returns:
            Elo rating or None if not found
        """
        try:
            elo_service = self.elo_service
            if elo_service is not None:
                rating = elo_service.get_rating(
                    participant_id=agent_id,
                    board_type=self.board_type,
                    num_players=self.num_players,
                )
                return rating.rating if rating else None

            rating = self.elo_db.get_rating(
                agent_id=agent_id,
                board_type=self.board_type,
                num_players=self.num_players,
            )
            return rating.elo if rating else None
        except Exception as e:
            logger.warning(f"Failed to get Elo for {agent_id}: {e}")
            return None

    def get_leaderboard(self, top_n: int = 10) -> list[dict[str, Any]]:
        """Get the current Elo leaderboard.

        Args:
            top_n: Number of top agents to return

        Returns:
            List of agent ratings sorted by Elo
        """
        try:
            elo_service = self.elo_service
            if elo_service is not None:
                entries = elo_service.get_leaderboard(
                    board_type=self.board_type,
                    num_players=self.num_players,
                    limit=top_n,
                )
                confidence_games = getattr(elo_service, "CONFIDENCE_GAMES", 0) or 1
                return [
                    {
                        "agent_id": entry.participant_id,
                        "elo": entry.rating,
                        "uncertainty": max(0.0, 1.0 - min(1.0, entry.games_played / confidence_games)),
                        "games_played": entry.games_played,
                    }
                    for entry in entries
                ]

            ratings = self.elo_db.get_ratings_for_config(
                board_type=self.board_type,
                num_players=self.num_players,
            )
            sorted_ratings = sorted(ratings, key=lambda r: r.elo, reverse=True)
            return [
                {
                    "agent_id": r.agent_id,
                    "elo": r.elo,
                    "uncertainty": r.uncertainty,
                    "games_played": r.games_played,
                }
                for r in sorted_ratings[:top_n]
            ]
        except Exception as e:
            logger.warning(f"Failed to get leaderboard: {e}")
            return []

    def _compute_confidence(self, games: int, elo_delta: float) -> float:
        """Compute confidence in the evaluation result.

        Simple heuristic based on game count and Elo magnitude.
        """
        # More games = more confidence
        game_factor = min(1.0, games / 100)

        # Larger Elo differences = more clear-cut
        elo_factor = min(1.0, abs(elo_delta) / 100)

        return (game_factor + elo_factor) / 2

    def _record_tournament_metrics(
        self,
        tournament_type: str,
        games: int,
        duration: float,
    ) -> None:
        """Record tournament metrics."""
        try:
            from app.metrics import record_evaluation
            record_evaluation(
                board_type=self.board_type,
                num_players=self.num_players,
                games=games,
                elo_delta=0,  # Aggregate, no single delta
                duration_seconds=duration,
                eval_type=tournament_type,
            )
        except Exception as e:
            logger.debug(f"Failed to record tournament metrics: {e}")

    def _record_evaluation_metrics(self, result: EvaluationResult) -> None:
        """Record evaluation metrics."""
        try:
            from app.metrics import record_evaluation
            record_evaluation(
                board_type=self.board_type,
                num_players=self.num_players,
                games=result.games_played,
                elo_delta=result.elo_delta,
                win_rate=result.win_rate,
                candidate_model=result.candidate_id,
                opponent=result.baseline_id,
            )
        except Exception as e:
            logger.debug(f"Failed to record evaluation metrics: {e}")


# =============================================================================
# Module-level convenience functions
# =============================================================================


def run_quick_evaluation(
    candidate: str,
    baselines: list[str] | None = None,
    board_type: str = "square8",
    num_players: int = 2,
    games_per_pairing: int = 20,
) -> tuple[bool, float]:
    """Run a quick model evaluation.

    Args:
        candidate: Candidate model ID
        baselines: Baseline models (defaults to standard baselines)
        board_type: Board type
        num_players: Number of players
        games_per_pairing: Games per baseline

    Returns:
        Tuple of (passed, average_elo_delta)
    """
    if baselines is None:
        baselines = ["random", "heuristic"]

    orchestrator = TournamentOrchestrator(
        board_type=board_type,
        num_players=num_players,
    )

    _, results = orchestrator.run_evaluation(
        candidate_model=candidate,
        baseline_models=baselines,
        games_per_pairing=games_per_pairing,
    )

    passed = all(r.passed for r in results)
    avg_delta = sum(r.elo_delta for r in results) / len(results) if results else 0

    return passed, avg_delta


def run_elo_calibration(
    agents: list[str],
    board_type: str = "square8",
    num_players: int = 2,
    games_per_pairing: int = 50,
) -> dict[str, float]:
    """Run an Elo calibration tournament.

    Args:
        agents: Agent IDs to calibrate
        board_type: Board type
        num_players: Number of players
        games_per_pairing: Games per pairing

    Returns:
        Dict mapping agent_id to Elo rating
    """
    orchestrator = TournamentOrchestrator(
        board_type=board_type,
        num_players=num_players,
    )

    summary = orchestrator.run_round_robin(
        agents=agents,
        games_per_pairing=games_per_pairing,
    )

    return summary.final_ratings
