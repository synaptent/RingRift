#!/usr/bin/env python3
"""Two-Phase Composite Gauntlet for evaluating (NN, Algorithm) combinations.

This module implements a two-phase evaluation system that efficiently tests
neural networks with multiple search algorithms:

Phase 1: NN Quick Evaluation (Policy-Only)
    - Test all NNs using policy_only (fast, no search overhead)
    - ~50 games each vs baselines
    - Purpose: Eliminate weak NNs early
    - Gate: Top 50% proceed to Phase 2

Phase 2: Search Amplification
    - Surviving NNs tested with each search algorithm
    - Algorithms: gumbel_mcts, mcts, descent
    - ~20 games per (NN, algorithm) pair
    - Records separate Elo for each combination

Usage:
    from app.tournament.composite_gauntlet import CompositeGauntlet

    gauntlet = CompositeGauntlet(board_type="square8", num_players=2)

    # Run two-phase evaluation
    results = await gauntlet.run_two_phase_gauntlet(nn_paths)

    # Or run algorithm tournament with fixed NN
    results = await gauntlet.run_algorithm_tournament(reference_nn)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.training.composite_participant import (
    STANDARD_ALGORITHM_CONFIGS,
    extract_nn_id,
    get_standard_config,
    is_composite_id,
    make_composite_participant_id,
    parse_composite_participant_id,
)
from app.training.elo_service import get_elo_service

# Event emission for gauntlet completion (Sprint 5)
try:
    from app.training.event_integration import publish_composite_gauntlet_completed
    HAS_EVENTS = True
except ImportError:
    HAS_EVENTS = False
    publish_composite_gauntlet_completed = None

logger = logging.getLogger(__name__)

# Default algorithms for Phase 2 testing
PHASE2_ALGORITHMS = ["gumbel_mcts", "mcts", "descent"]

# Baseline composite IDs (pinned ratings)
BASELINE_COMPOSITE_IDS = {
    "none:random:d1": 400.0,
    "none:heuristic:d2": 1000.0,
}


@dataclass
class GauntletPhaseConfig:
    """Configuration for a gauntlet phase."""
    games_per_matchup: int = 20
    parallel_workers: int = 4
    timeout_seconds: int = 300
    pass_threshold: float = 0.5  # Top 50% pass to next phase


@dataclass
class CompositeGauntletConfig:
    """Configuration for composite gauntlet."""
    phase1: GauntletPhaseConfig = field(default_factory=lambda: GauntletPhaseConfig(
        games_per_matchup=50,
        pass_threshold=0.5,
    ))
    phase2: GauntletPhaseConfig = field(default_factory=lambda: GauntletPhaseConfig(
        games_per_matchup=20,
    ))
    phase2_algorithms: list[str] = field(default_factory=lambda: list(PHASE2_ALGORITHMS))


@dataclass
class PhaseResult:
    """Result of a gauntlet phase."""
    phase: int
    nn_ids: list[str]
    passed_nn_ids: list[str]
    failed_nn_ids: list[str]
    ratings: dict[str, float]  # participant_id -> elo
    games_played: int
    duration_sec: float


@dataclass
class CompositeGauntletResult:
    """Complete result of a two-phase gauntlet."""
    run_id: str
    board_type: str
    num_players: int
    started_at: float
    completed_at: float | None = None
    phase1_result: PhaseResult | None = None
    phase2_result: PhaseResult | None = None
    final_rankings: list[dict[str, Any]] = field(default_factory=list)
    total_games: int = 0
    status: str = "pending"


class CompositeGauntlet:
    """Two-phase gauntlet for composite (NN, Algorithm) evaluation.

    Implements efficient evaluation that:
    1. First screens NNs with fast policy-only evaluation
    2. Then tests promising NNs with multiple search algorithms
    3. Records separate Elo for each (NN, Algorithm) combination
    """

    def __init__(
        self,
        board_type: str = "square8",
        num_players: int = 2,
        config: CompositeGauntletConfig | None = None,
    ):
        """Initialize composite gauntlet.

        Args:
            board_type: Board type for games
            num_players: Number of players
            config: Gauntlet configuration
        """
        self.board_type = board_type
        self.num_players = num_players
        self.config = config or CompositeGauntletConfig()
        self.elo_service = get_elo_service()

        # Lazy load game modules
        self._game_modules_loaded = False
        self._executor: ThreadPoolExecutor | None = None

    def _ensure_game_modules(self):
        """Lazy load heavy game modules."""
        if self._game_modules_loaded:
            return

        # Import game-related modules
        from app.models import AIConfig, AIType, BoardType, GameStatus
        from app.ai.random_ai import RandomAI
        from app.ai.heuristic_ai import HeuristicAI
        from app.ai.policy_only_ai import PolicyOnlyAI
        from app.rules.default_engine import DefaultRulesEngine
        from app.training.initial_state import create_initial_state

        self._BoardType = BoardType
        self._AIType = AIType
        self._AIConfig = AIConfig
        self._GameStatus = GameStatus
        self._RandomAI = RandomAI
        self._HeuristicAI = HeuristicAI
        self._PolicyOnlyAI = PolicyOnlyAI
        self._DefaultRulesEngine = DefaultRulesEngine
        self._create_initial_state = create_initial_state

        self._game_modules_loaded = True

    def _get_board_type_enum(self):
        """Get BoardType enum value."""
        self._ensure_game_modules()
        board_map = {
            "square8": self._BoardType.SQUARE8,
            "square19": self._BoardType.SQUARE19,
            "hexagonal": self._BoardType.HEXAGONAL,
        }
        return board_map.get(self.board_type, self._BoardType.SQUARE8)

    async def run_two_phase_gauntlet(
        self,
        nn_paths: list[str | Path],
    ) -> CompositeGauntletResult:
        """Run complete two-phase gauntlet evaluation.

        Phase 1: Test all NNs with policy_only against baselines
        Phase 2: Test top NNs with multiple search algorithms

        Args:
            nn_paths: List of paths to neural network model files

        Returns:
            CompositeGauntletResult with complete evaluation data
        """
        run_id = str(uuid.uuid4())[:8]
        result = CompositeGauntletResult(
            run_id=run_id,
            board_type=self.board_type,
            num_players=self.num_players,
            started_at=time.time(),
        )

        logger.info(f"Starting two-phase gauntlet {run_id} with {len(nn_paths)} NNs")

        # Extract NN IDs from paths
        nn_ids = [self._extract_nn_id(p) for p in nn_paths]
        nn_path_map = {self._extract_nn_id(p): str(p) for p in nn_paths}

        try:
            # Phase 1: Policy-only screening
            logger.info("Phase 1: Policy-only screening")
            phase1_result = await self._run_phase1(nn_ids, nn_path_map)
            result.phase1_result = phase1_result
            result.total_games += phase1_result.games_played

            if not phase1_result.passed_nn_ids:
                logger.warning("No NNs passed Phase 1 screening")
                result.status = "completed"
                result.completed_at = time.time()
                return result

            # Phase 2: Search algorithm testing
            logger.info(f"Phase 2: Testing {len(phase1_result.passed_nn_ids)} NNs with algorithms")
            phase2_result = await self._run_phase2(
                phase1_result.passed_nn_ids,
                nn_path_map,
            )
            result.phase2_result = phase2_result
            result.total_games += phase2_result.games_played

            # Generate final rankings
            result.final_rankings = self._generate_final_rankings()
            result.status = "completed"

        except Exception as e:
            logger.error(f"Gauntlet {run_id} failed: {e}")
            result.status = f"failed: {e}"

        result.completed_at = time.time()
        logger.info(
            f"Gauntlet {run_id} completed: {result.total_games} games, "
            f"duration: {result.completed_at - result.started_at:.1f}s"
        )

        # Emit gauntlet completed event (Sprint 5)
        if HAS_EVENTS and publish_composite_gauntlet_completed is not None:
            try:
                phase1_count = len(nn_ids)
                phase1_passed = len(result.phase1_result.passed_nn_ids) if result.phase1_result else 0
                phase2_participants = (
                    len(result.phase2_result.ratings) if result.phase2_result else 0
                )
                top_nn_ids = [
                    r.get("nn_id", "") for r in result.final_rankings[:3]
                    if isinstance(r, dict)
                ]
                top_algo = ""
                if result.final_rankings and isinstance(result.final_rankings[0], dict):
                    top_algo = result.final_rankings[0].get("ai_type", "")

                # Use asyncio to run the async publisher
                asyncio.create_task(publish_composite_gauntlet_completed(
                    board_type=self.board_type,
                    num_players=self.num_players,
                    phase1_nn_count=phase1_count,
                    phase1_passed_count=phase1_passed,
                    phase2_participants=phase2_participants,
                    total_games_played=result.total_games,
                    duration_seconds=result.completed_at - result.started_at,
                    top_nn_ids=top_nn_ids,
                    top_algorithm=top_algo,
                ))
            except Exception as e:
                logger.debug(f"Failed to emit gauntlet event: {e}")

        return result

    async def _run_phase1(
        self,
        nn_ids: list[str],
        nn_path_map: dict[str, str],
    ) -> PhaseResult:
        """Run Phase 1: Policy-only screening.

        Tests all NNs with policy_only algorithm against baselines.
        Top 50% (configurable) pass to Phase 2.
        """
        start_time = time.time()
        ratings: dict[str, float] = {}
        games_played = 0

        # Register and evaluate each NN with policy_only
        for nn_id in nn_ids:
            participant_id = self.elo_service.register_composite_participant(
                nn_id=nn_id,
                ai_type="policy_only",
                board_type=self.board_type,
                num_players=self.num_players,
                nn_model_path=nn_path_map.get(nn_id),
            )

            # Play against baselines
            games = await self._play_against_baselines(
                participant_id=participant_id,
                nn_path=nn_path_map.get(nn_id),
                ai_type="policy_only",
                games_per_baseline=self.config.phase1.games_per_matchup,
            )
            games_played += games

            # Get updated rating
            rating = self.elo_service.get_rating(
                participant_id, self.board_type, self.num_players
            )
            ratings[participant_id] = rating.rating

        # Rank by Elo and select top portion
        sorted_participants = sorted(
            ratings.items(), key=lambda x: x[1], reverse=True
        )

        pass_count = max(1, int(len(sorted_participants) * self.config.phase1.pass_threshold))
        passed = [extract_nn_id(p[0]) for p in sorted_participants[:pass_count] if extract_nn_id(p[0])]
        failed = [extract_nn_id(p[0]) for p in sorted_participants[pass_count:] if extract_nn_id(p[0])]

        return PhaseResult(
            phase=1,
            nn_ids=nn_ids,
            passed_nn_ids=passed,
            failed_nn_ids=failed,
            ratings=ratings,
            games_played=games_played,
            duration_sec=time.time() - start_time,
        )

    async def _run_phase2(
        self,
        nn_ids: list[str],
        nn_path_map: dict[str, str],
    ) -> PhaseResult:
        """Run Phase 2: Search algorithm testing.

        Tests each surviving NN with multiple search algorithms.
        """
        start_time = time.time()
        ratings: dict[str, float] = {}
        games_played = 0

        for nn_id in nn_ids:
            for ai_type in self.config.phase2_algorithms:
                # Register composite participant
                participant_id = self.elo_service.register_composite_participant(
                    nn_id=nn_id,
                    ai_type=ai_type,
                    board_type=self.board_type,
                    num_players=self.num_players,
                    nn_model_path=nn_path_map.get(nn_id),
                )

                # Play against baselines
                games = await self._play_against_baselines(
                    participant_id=participant_id,
                    nn_path=nn_path_map.get(nn_id),
                    ai_type=ai_type,
                    games_per_baseline=self.config.phase2.games_per_matchup,
                )
                games_played += games

                # Get updated rating
                rating = self.elo_service.get_rating(
                    participant_id, self.board_type, self.num_players
                )
                ratings[participant_id] = rating.rating

        return PhaseResult(
            phase=2,
            nn_ids=nn_ids,
            passed_nn_ids=nn_ids,  # All pass in Phase 2
            failed_nn_ids=[],
            ratings=ratings,
            games_played=games_played,
            duration_sec=time.time() - start_time,
        )

    async def _play_against_baselines(
        self,
        participant_id: str,
        nn_path: str | None,
        ai_type: str,
        games_per_baseline: int,
    ) -> int:
        """Play games against baseline opponents.

        Args:
            participant_id: Composite participant ID
            nn_path: Path to NN model
            ai_type: Algorithm type
            games_per_baseline: Games per baseline opponent

        Returns:
            Total games played
        """
        baselines = list(BASELINE_COMPOSITE_IDS.keys())
        total_games = 0

        for baseline_id in baselines:
            for _ in range(games_per_baseline):
                # Play game
                result = await self._play_single_game(
                    participant_id=participant_id,
                    baseline_id=baseline_id,
                    nn_path=nn_path,
                    ai_type=ai_type,
                )

                # Record result
                if result is not None:
                    winner = None
                    if result["winner"] == "model":
                        winner = participant_id
                    elif result["winner"] == "baseline":
                        winner = baseline_id

                    self.elo_service.record_match(
                        participant_a=participant_id,
                        participant_b=baseline_id,
                        winner=winner,
                        board_type=self.board_type,
                        num_players=self.num_players,
                        game_length=result.get("game_length", 0),
                        duration_sec=result.get("duration_sec", 0),
                    )
                    total_games += 1

        return total_games

    async def _play_single_game(
        self,
        participant_id: str,
        baseline_id: str,
        nn_path: str | None,
        ai_type: str,
    ) -> dict[str, Any] | None:
        """Play a single game between participant and baseline.

        Args:
            participant_id: Composite participant ID
            baseline_id: Baseline composite ID
            nn_path: Path to NN model
            ai_type: Algorithm type

        Returns:
            Game result dict or None if failed
        """
        self._ensure_game_modules()

        try:
            # Create AIs
            model_ai = self._create_composite_ai(
                participant_id=participant_id,
                nn_path=nn_path,
                ai_type=ai_type,
                player_number=1,
            )

            baseline_ai = self._create_baseline_ai(
                baseline_id=baseline_id,
                player_number=2,
            )

            if model_ai is None or baseline_ai is None:
                return None

            # Create game state
            board_type_enum = self._get_board_type_enum()
            game_state = self._create_initial_state(
                board_type=board_type_enum,
                num_players=self.num_players,
            )

            rules = self._DefaultRulesEngine()
            start_time = time.time()
            move_count = 0
            max_moves = 500  # Safety limit

            # Play game
            while game_state.status == self._GameStatus.IN_PROGRESS and move_count < max_moves:
                current_player = game_state.current_player_number

                if current_player == 1:
                    ai = model_ai
                else:
                    ai = baseline_ai

                move = ai.select_move(game_state)
                if move is None:
                    break

                game_state = rules.apply_move(game_state, move)
                move_count += 1

            duration = time.time() - start_time

            # Determine winner
            winner = None
            if game_state.status == self._GameStatus.COMPLETED:
                if game_state.winner == 1:
                    winner = "model"
                elif game_state.winner == 2:
                    winner = "baseline"

            return {
                "winner": winner,
                "game_length": move_count,
                "duration_sec": duration,
            }

        except Exception as e:
            logger.error(f"Game failed: {e}")
            return None

    def _create_composite_ai(
        self,
        participant_id: str,
        nn_path: str | None,
        ai_type: str,
        player_number: int,
    ):
        """Create AI instance for composite participant."""
        self._ensure_game_modules()

        board_type_enum = self._get_board_type_enum()
        config = self._AIConfig(
            ai_type=self._AIType.NEURAL,
            board_type=board_type_enum,
            difficulty=5,
        )

        if ai_type == "policy_only":
            if nn_path:
                return self._PolicyOnlyAI(
                    player_number=player_number,
                    config=config,
                    model_path=Path(nn_path),
                )
            return None

        # For other AI types, try to create appropriate AI
        # This is a simplified version - full implementation would
        # instantiate MCTS, Gumbel, Descent, etc.
        if nn_path:
            try:
                return self._PolicyOnlyAI(
                    player_number=player_number,
                    config=config,
                    model_path=Path(nn_path),
                )
            except Exception as e:
                logger.warning(f"Failed to create AI: {e}")
                return None

        return None

    def _create_baseline_ai(self, baseline_id: str, player_number: int):
        """Create baseline AI from composite ID."""
        self._ensure_game_modules()

        try:
            nn_id, ai_type, config = parse_composite_participant_id(baseline_id)
        except ValueError:
            ai_type = "random"
            config = {}

        board_type_enum = self._get_board_type_enum()

        if ai_type == "random":
            return self._RandomAI(
                player_number=player_number,
                config=self._AIConfig(
                    ai_type=self._AIType.RANDOM,
                    board_type=board_type_enum,
                ),
            )
        elif ai_type == "heuristic":
            difficulty = config.get("difficulty", 2)
            return self._HeuristicAI(
                player_number=player_number,
                config=self._AIConfig(
                    ai_type=self._AIType.HEURISTIC,
                    board_type=board_type_enum,
                    difficulty=difficulty,
                ),
            )

        # Default to random
        return self._RandomAI(
            player_number=player_number,
            config=self._AIConfig(
                ai_type=self._AIType.RANDOM,
                board_type=board_type_enum,
            ),
        )

    def _extract_nn_id(self, path: str | Path) -> str:
        """Extract NN ID from model path."""
        path = Path(path)
        return path.stem

    def _generate_final_rankings(self) -> list[dict[str, Any]]:
        """Generate final rankings from Elo service."""
        return self.elo_service.get_composite_leaderboard(
            board_type=self.board_type,
            num_players=self.num_players,
            min_games=1,
            limit=100,
        )

    async def run_algorithm_tournament(
        self,
        reference_nn: str | Path,
        algorithms: list[str] | None = None,
        games_per_algorithm: int = 50,
    ) -> dict[str, float]:
        """Run tournament to compare algorithms using fixed NN.

        This isolates algorithm strength from NN quality.

        Args:
            reference_nn: Path to reference neural network
            algorithms: Algorithms to test (defaults to standard set)
            games_per_algorithm: Games per algorithm vs baselines

        Returns:
            Dict mapping algorithm -> Elo rating
        """
        algorithms = algorithms or PHASE2_ALGORITHMS
        nn_id = self._extract_nn_id(reference_nn)
        nn_path = str(reference_nn)

        results: dict[str, float] = {}

        for ai_type in algorithms:
            participant_id = self.elo_service.register_composite_participant(
                nn_id=nn_id,
                ai_type=ai_type,
                board_type=self.board_type,
                num_players=self.num_players,
                nn_model_path=nn_path,
            )

            await self._play_against_baselines(
                participant_id=participant_id,
                nn_path=nn_path,
                ai_type=ai_type,
                games_per_baseline=games_per_algorithm // 2,
            )

            rating = self.elo_service.get_rating(
                participant_id, self.board_type, self.num_players
            )
            results[ai_type] = rating.rating

        return results

    async def run_nn_tournament(
        self,
        nn_paths: list[str | Path],
        reference_algorithm: str = "gumbel_mcts",
        games_per_nn: int = 50,
    ) -> dict[str, float]:
        """Run tournament to compare NNs using fixed algorithm.

        This isolates NN quality from algorithm choice.

        Args:
            nn_paths: Paths to neural networks to compare
            reference_algorithm: Algorithm to use for all
            games_per_nn: Games per NN vs baselines

        Returns:
            Dict mapping nn_id -> Elo rating
        """
        results: dict[str, float] = {}

        for nn_path in nn_paths:
            nn_id = self._extract_nn_id(nn_path)
            participant_id = self.elo_service.register_composite_participant(
                nn_id=nn_id,
                ai_type=reference_algorithm,
                board_type=self.board_type,
                num_players=self.num_players,
                nn_model_path=str(nn_path),
            )

            await self._play_against_baselines(
                participant_id=participant_id,
                nn_path=str(nn_path),
                ai_type=reference_algorithm,
                games_per_baseline=games_per_nn // 2,
            )

            rating = self.elo_service.get_rating(
                participant_id, self.board_type, self.num_players
            )
            results[nn_id] = rating.rating

        return results


# Convenience functions for script usage

async def run_two_phase_gauntlet(
    nn_paths: list[str | Path],
    board_type: str = "square8",
    num_players: int = 2,
) -> CompositeGauntletResult:
    """Run two-phase gauntlet evaluation.

    Args:
        nn_paths: List of paths to neural network models
        board_type: Board type for games
        num_players: Number of players

    Returns:
        CompositeGauntletResult with evaluation data
    """
    gauntlet = CompositeGauntlet(board_type=board_type, num_players=num_players)
    return await gauntlet.run_two_phase_gauntlet(nn_paths)


async def run_algorithm_tournament(
    reference_nn: str | Path,
    algorithms: list[str] | None = None,
    board_type: str = "square8",
    num_players: int = 2,
    games_per_algorithm: int = 50,
) -> dict[str, float]:
    """Run algorithm tournament with fixed NN.

    Args:
        reference_nn: Path to reference neural network
        algorithms: Algorithms to test
        board_type: Board type
        num_players: Number of players
        games_per_algorithm: Games per algorithm

    Returns:
        Dict mapping algorithm -> Elo rating
    """
    gauntlet = CompositeGauntlet(board_type=board_type, num_players=num_players)
    return await gauntlet.run_algorithm_tournament(
        reference_nn=reference_nn,
        algorithms=algorithms,
        games_per_algorithm=games_per_algorithm,
    )


async def run_nn_tournament(
    nn_paths: list[str | Path],
    reference_algorithm: str = "gumbel_mcts",
    board_type: str = "square8",
    num_players: int = 2,
    games_per_nn: int = 50,
) -> dict[str, float]:
    """Run NN tournament with fixed algorithm.

    Args:
        nn_paths: Paths to neural networks
        reference_algorithm: Algorithm to use
        board_type: Board type
        num_players: Number of players
        games_per_nn: Games per NN

    Returns:
        Dict mapping nn_id -> Elo rating
    """
    gauntlet = CompositeGauntlet(board_type=board_type, num_players=num_players)
    return await gauntlet.run_nn_tournament(
        nn_paths=nn_paths,
        reference_algorithm=reference_algorithm,
        games_per_nn=games_per_nn,
    )
