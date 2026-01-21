"""Tournament match execution and result aggregation."""
from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.models import BoardType, GameStatus
from app.training.composite_participant import extract_harness_type

from .agents import AgentType, AIAgent, AIAgentRegistry
from .elo import EloCalculator
from .recording import TournamentRecordingOptions
from .scheduler import Match, TournamentScheduler

logger = logging.getLogger(__name__)


def _exponential_backoff_delay(
    attempt: int,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
) -> float:
    """Calculate exponential backoff delay with jitter.

    Args:
        attempt: Zero-indexed attempt number
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Delay in seconds with jitter applied
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = delay * 0.2 * random.random()
    return delay + jitter


def extract_model_metadata(model_path: str) -> dict[str, Any]:
    """Extract comprehensive metadata from a model checkpoint.

    Returns a dict with:
        - model_type: "nnue", "cnn", or "unknown"
        - model_class: Class name if available
        - architecture_version: Version string if available
        - board_type: Inferred board type if available
        - num_players: Inferred player count if available
        - created_at: Checkpoint creation timestamp if available
        - training_config: Training configuration if available
    """
    result: dict[str, Any] = {
        "model_type": "unknown",
        "model_class": None,
        "architecture_version": None,
        "board_type": None,
        "num_players": None,
        "created_at": None,
        "training_config": None,
    }

    try:
        from app.utils.torch_utils import safe_load_checkpoint

        checkpoint = safe_load_checkpoint(model_path, map_location="cpu", warn_on_unsafe=False)

        if not isinstance(checkpoint, dict):
            return result

        # Extract from versioning metadata
        meta = checkpoint.get("_versioning_metadata")
        if isinstance(meta, dict):
            result["model_class"] = meta.get("model_class")
            result["architecture_version"] = meta.get("architecture_version")
            result["created_at"] = meta.get("created_at")

            # Check for model class to infer type
            model_class = meta.get("model_class", "")
            if isinstance(model_class, str):
                lower_class = model_class.lower()
                if "nnue" in lower_class:
                    result["model_type"] = "nnue"
                elif "cnn" in lower_class or "hexneuralnet" in lower_class:
                    result["model_type"] = "cnn"

            # Extract board_type and num_players from config
            config = meta.get("config", {})
            if isinstance(config, dict):
                result["board_type"] = config.get("board_type")
                result["num_players"] = config.get("num_players")
                result["training_config"] = config

        # Check direct keys for board_type/num_players
        if result["board_type"] is None and "board_type" in checkpoint:
            result["board_type"] = checkpoint["board_type"]
        if result["num_players"] is None and "num_players" in checkpoint:
            result["num_players"] = checkpoint["num_players"]

        # Infer model type from state_dict keys if not yet determined
        if result["model_type"] == "unknown":
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
            if isinstance(state_dict, dict):
                keys = set(state_dict.keys())
                if "accumulator.weight" in keys:
                    result["model_type"] = "nnue"
                elif "conv1.weight" in keys or any("res_blocks" in k for k in keys) or "hex_mask" in keys:
                    result["model_type"] = "cnn"

        # Infer board_type from filename if not in metadata
        if result["board_type"] is None:
            path_lower = model_path.lower()
            if "sq8" in path_lower or "square8" in path_lower:
                result["board_type"] = "square8"
            elif "sq19" in path_lower or "square19" in path_lower:
                result["board_type"] = "square19"
            elif "hex" in path_lower:
                result["board_type"] = "hexagonal"

        # Infer num_players from filename if not in metadata
        if result["num_players"] is None:
            import re
            match = re.search(r"_(\d)p[_\.]", model_path.lower())
            if match:
                result["num_players"] = int(match.group(1))
            elif "_2p" in model_path.lower():
                result["num_players"] = 2
            elif "_3p" in model_path.lower():
                result["num_players"] = 3
            elif "_4p" in model_path.lower():
                result["num_players"] = 4

        return result

    except Exception as e:
        logger.warning(f"Could not extract metadata from {model_path}: {e}")
        return result


def detect_model_type(model_path: str) -> str:
    """Detect the model type from a checkpoint file.

    Returns one of: "nnue", "cnn", or "unknown"

    This is a convenience wrapper around extract_model_metadata().
    """
    return extract_model_metadata(model_path).get("model_type", "unknown")


@dataclass
class MatchResult:
    """Result of a single match."""

    match_id: str
    agent_ids: list[str]
    rankings: list[str]  # Ordered by finish position (1st, 2nd, ...)
    winner: str | None  # Agent ID of winner, None for draw/timeout
    game_length: int
    termination_reason: str
    duration_seconds: float
    metadata: dict = field(default_factory=dict)

    @property
    def is_draw(self) -> bool:
        return self.winner is None and "draw" in self.termination_reason.lower()

    def to_dict(self) -> dict:
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
    completed_at: datetime | None = None
    match_results: list[MatchResult] = field(default_factory=list)
    final_ratings: dict[str, float] = field(default_factory=dict)
    agent_stats: dict[str, dict] = field(default_factory=dict)

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
        for _agent_id, stats in self.agent_stats.items():
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

    def to_dict(self) -> dict:
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
        elo_calculator: EloCalculator | None = None,
        max_workers: int = 4,
        max_moves: int = 10000,
        seed: int | None = None,
        persist_to_unified_elo: bool = True,
        tournament_id: str | None = None,
        recording_options: TournamentRecordingOptions | None = None,
        match_retries: int = 2,
    ):
        """Initialize tournament runner.

        Args:
            agent_registry: Registry of AI agents.
            scheduler: Match scheduler.
            elo_calculator: Elo rating calculator (created if None).
            max_workers: Maximum parallel workers for local execution.
            max_moves: Maximum moves per game before timeout.
            seed: Random seed for reproducibility.
            persist_to_unified_elo: If True, persist results to unified Elo database.
            tournament_id: Optional tournament ID for tracking.
            recording_options: Optional recording configuration for canonical replay data.
            match_retries: Number of retry attempts for failed matches (default: 2).
        """
        self.agent_registry = agent_registry
        self.scheduler = scheduler
        self.elo_calculator = elo_calculator or EloCalculator()
        self.max_workers = max_workers
        self.max_moves = max_moves
        self.seed = seed
        self._rng = random.Random(seed)
        self.persist_to_unified_elo = persist_to_unified_elo
        self.tournament_id = tournament_id
        self.recording_options = recording_options or TournamentRecordingOptions()
        self.match_retries = match_retries

        self.results: TournamentResults | None = None
        self._match_executor: Callable | None = None
        self._elo_service = None
        self._unified_elo_db = None

        # Prefer EloService for persistence; fallback to legacy unified_elo_db
        if self.persist_to_unified_elo:
            try:
                from app.training.elo_service import get_elo_service
                self._elo_service = get_elo_service()
            except ImportError:
                self._elo_service = None

            if self._elo_service is None:
                try:
                    from .unified_elo_db import get_elo_database
                    self._unified_elo_db = get_elo_database()
                except ImportError:
                    pass

    def set_match_executor(
        self,
        executor: Callable[[Match, dict[str, AIAgent]], MatchResult],
    ) -> None:
        """Set custom match executor for distributed execution.

        Args:
            executor: Function that takes (match, agents) and returns MatchResult.
        """
        self._match_executor = executor

    def run_tournament(
        self,
        agent_ids: list[str],
        board_type: BoardType,
        num_players: int = 2,
        games_per_pairing: int = 2,
        progress_callback: Callable[[int, int], None] | None = None,
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
            # Use custom executor (for distributed execution) with retry logic
            for match in matches:
                result = None
                last_error = None
                for attempt in range(self.match_retries):
                    try:
                        result = self._match_executor(match, agents)
                        break  # Success
                    except Exception as e:
                        last_error = e
                        if attempt < self.match_retries - 1:
                            delay = _exponential_backoff_delay(attempt)
                            logger.warning(
                                f"Match {match.match_id} failed (attempt {attempt + 1}/{self.match_retries}), "
                                f"retrying in {delay:.1f}s: {e}"
                            )
                            time.sleep(delay)

                if result is not None:
                    self._process_result(match, result)
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total_matches)
                else:
                    logger.error(f"Match {match.match_id} failed after {self.match_retries} attempts: {last_error}")
                    self.scheduler.mark_match_failed(match.match_id, str(last_error))
        else:
            # Local parallel execution with retry support
            def execute_with_retries(match: Match) -> tuple[Match, MatchResult | None, Exception | None]:
                """Execute a match with retry logic."""
                last_error: Exception | None = None
                for attempt in range(self.match_retries):
                    try:
                        result = self._execute_match_local(match, agents)
                        return (match, result, None)
                    except Exception as e:
                        last_error = e
                        if attempt < self.match_retries - 1:
                            delay = _exponential_backoff_delay(attempt)
                            logger.warning(
                                f"Match {match.match_id} failed (attempt {attempt + 1}/{self.match_retries}), "
                                f"retrying in {delay:.1f}s: {e}"
                            )
                            time.sleep(delay)
                return (match, None, last_error)

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(execute_with_retries, match): match
                    for match in matches
                }

                for future in as_completed(futures):
                    match = futures[future]
                    try:
                        returned_match, result, error = future.result()
                        if result is not None:
                            self._process_result(returned_match, result)
                            completed += 1
                            if progress_callback:
                                progress_callback(completed, total_matches)
                        else:
                            logger.error(f"Match {returned_match.match_id} failed after {self.match_retries} attempts: {error}")
                            self.scheduler.mark_match_failed(returned_match.match_id, str(error))
                    except Exception as e:
                        logger.error(f"Match {match.match_id} execution error: {e}")
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

        # Persist to Elo service (preferred) or legacy unified_elo_db fallback.
        if self._elo_service is not None:
            try:
                board_type = match.board_type.value
                tournament_id = self.tournament_id or "default"
                # January 2026: Extract harness_type from agent IDs for per-harness Elo
                # Default to gumbel_mcts for legacy model names without composite ID
                harness_type = (extract_harness_type(result.agent_ids[0]) if result.agent_ids else None) or "gumbel_mcts"
                if len(result.rankings) == 2:
                    self._elo_service.record_match(
                        result.agent_ids[0],
                        result.agent_ids[1],
                        winner=result.winner,
                        board_type=board_type,
                        num_players=match.num_players,
                        game_length=result.game_length,
                        duration_sec=result.duration_seconds,
                        tournament_id=tournament_id,
                        harness_type=harness_type,
                    )
                else:
                    self._record_multiplayer_elo(
                        ordered_agent_ids=result.rankings,
                        board_type=board_type,
                        num_players=match.num_players,
                        tournament_id=tournament_id,
                        game_length=result.game_length,
                        duration_sec=result.duration_seconds,
                        harness_type=harness_type,
                    )
            except Exception as e:
                logger.warning(f"Failed to persist match to Elo service: {e}")
        elif self._unified_elo_db is not None:
            try:
                rankings = [
                    result.rankings.index(pid) if pid in result.rankings else 0
                    for pid in result.agent_ids
                ]
                self._unified_elo_db.record_match_and_update(
                    participant_ids=result.agent_ids,
                    rankings=rankings,
                    board_type=match.board_type.value,
                    num_players=match.num_players,
                    tournament_id=self.tournament_id or "default",
                    game_length=result.game_length,
                    duration_sec=result.duration_seconds,
                )
            except Exception as e:
                logger.warning(f"Failed to persist match to unified Elo DB: {e}")

        # Store result
        self.results.add_result(result)

    def _record_multiplayer_elo(
        self,
        ordered_agent_ids: list[str],
        board_type: str,
        num_players: int,
        tournament_id: str,
        game_length: int,
        duration_sec: float,
        harness_type: str | None = None,
    ) -> None:
        """Record a multiplayer result into EloService via pairwise matches.

        This decomposes the ranking into pairwise wins (higher-ranked beats
        lower-ranked). It is a pragmatic approximation until EloService has a
        native multiplayer update API.
        """
        if self._elo_service is None:
            return

        for i in range(len(ordered_agent_ids)):
            winner_id = ordered_agent_ids[i]
            for j in range(i + 1, len(ordered_agent_ids)):
                loser_id = ordered_agent_ids[j]
                self._elo_service.record_match(
                    winner_id,
                    loser_id,
                    winner=winner_id,
                    board_type=board_type,
                    num_players=num_players,
                    game_length=game_length,
                    duration_sec=duration_sec,
                    tournament_id=tournament_id,
                    harness_type=harness_type,
                )

    def _create_ai_instance(
        self,
        agent: AIAgent,
        board_type: BoardType,
        num_players: int,
    ) -> Any:
        """Create an AI instance based on agent type.

        Args:
            agent: The agent configuration
            board_type: Board type for the game
            num_players: Number of players in the game

        Returns:
            An AI instance that can provide moves via get_best_move()
        """
        # Convert BoardType to string for APIs that expect string
        board_type_str = board_type.value if hasattr(board_type, 'value') else str(board_type)

        if agent.agent_type == AgentType.NEURAL:
            # Detect model type to choose the right agent class
            model_type = detect_model_type(agent.model_path)
            logger.info(f"Detected model type '{model_type}' for agent {agent.agent_id}")

            if model_type == "nnue":
                try:
                    from app.ai.nnue_policy import NNUEPolicyAgent
                    return NNUEPolicyAgent(
                        model_path=agent.model_path,
                        board_type=board_type_str,
                        num_players=num_players,
                    )
                except Exception as e:
                    logger.warning(f"Failed to load NNUE agent: {e}")

            elif model_type == "cnn":
                try:
                    from pathlib import Path

                    from app.ai.neural_net import NeuralNetAI
                    from app.models import AIConfig

                    # Extract model ID from path (e.g., "models/ringrift_hex8_2p.pth" -> "ringrift_hex8_2p")
                    model_id = Path(agent.model_path).stem

                    # Create config for NeuralNetAI
                    config = AIConfig(
                        difficulty=5,  # Default difficulty
                        nn_model_id=model_id,
                        allow_fresh_weights=False,
                    )

                    # Create CNN agent and wrap to adapt interface
                    # player_number=0 since we're using policy network, not turn-based eval
                    neural_ai = NeuralNetAI(player_number=0, config=config, board_type=board_type)

                    # Wrapper to adapt select_move to get_best_move interface
                    class CNNAgentWrapper:
                        def __init__(self, neural_ai):
                            self._ai = neural_ai

                        def get_best_move(self, state, legal_moves):
                            # NeuralNetAI.select_move handles legal moves internally
                            return self._ai.select_move(state)

                    return CNNAgentWrapper(neural_ai)
                except Exception as e:
                    logger.warning(f"Failed to load CNN agent: {e}")
                    import traceback
                    traceback.print_exc()

            # Fallback to heuristic if neural loading fails
            logger.warning(
                f"Could not load neural model for {agent.agent_id}, falling back to heuristic"
            )
            return self._create_heuristic_agent(agent)

        elif agent.agent_type == AgentType.RANDOM:
            # Random agent - returns random legal move
            class RandomAgent:
                def get_best_move(self, state, legal_moves):
                    import random
                    return random.choice(legal_moves) if legal_moves else None
            return RandomAgent()

        elif agent.agent_type == AgentType.HEURISTIC:
            return self._create_heuristic_agent(agent)

        elif agent.agent_type == AgentType.MINIMAX:
            try:
                from app.ai.minimax_ai import MinimaxAI
                from app.models import AIConfig, AIType

                difficulty = self._resolve_difficulty(agent, default_offset=2)
                max_depth = max(1, int(agent.search_depth or 3))
                config = AIConfig(
                    ai_type=AIType.MINIMAX,
                    difficulty=difficulty,
                    max_depth=max_depth,
                )
                minimax_ai = MinimaxAI(player_number=0, config=config)

                class MinimaxAgentWrapper:
                    def __init__(self, ai):
                        self._ai = ai

                    def get_best_move(self, state, legal_moves):
                        return self._ai.select_move(state)

                return MinimaxAgentWrapper(minimax_ai)
            except ImportError as e:
                logger.warning(f"MinimaxAI not available: {e}")
                return self._create_heuristic_agent(agent)

        elif agent.agent_type == AgentType.MCTS:
            # MCTS agent
            try:
                from app.ai.mcts_ai import MCTSAI
                from app.models import AIConfig

                difficulty = self._resolve_difficulty(agent, default_offset=2)
                config = AIConfig(
                    difficulty=difficulty,
                    mcts_iterations=agent.mcts_simulations,
                )
                mcts_ai = MCTSAI(player_number=0, config=config)

                # Wrapper to adapt select_move to get_best_move interface
                class MCTSAgentWrapper:
                    def __init__(self, ai):
                        self._ai = ai

                    def get_best_move(self, state, legal_moves):
                        return self._ai.select_move(state)

                return MCTSAgentWrapper(mcts_ai)
            except ImportError as e:
                logger.warning(f"MCTSAI not available: {e}")
                return self._create_heuristic_agent(agent)

        elif agent.agent_type == AgentType.DESCENT:
            try:
                from app.ai.descent_ai import DescentAI
                from app.models import AIConfig, AIType

                difficulty = self._resolve_difficulty(agent, default_offset=2)
                config = AIConfig(
                    ai_type=AIType.DESCENT,
                    difficulty=difficulty,
                )
                descent_ai = DescentAI(player_number=0, config=config)

                class DescentAgentWrapper:
                    def __init__(self, ai):
                        self._ai = ai

                    def get_best_move(self, state, legal_moves):
                        return self._ai.select_move(state)

                return DescentAgentWrapper(descent_ai)
            except ImportError as e:
                logger.warning(f"DescentAI not available: {e}")
                return self._create_heuristic_agent(agent)

        else:
            # Default to heuristic (minimax-like)
            return self._create_heuristic_agent(agent)

    def _create_heuristic_agent(self, agent: AIAgent) -> Any:
        """Create a heuristic-based AI agent.

        Uses HeuristicAI which provides a simple evaluation-based move selector.
        """
        from app.ai.heuristic_ai import HeuristicAI
        from app.models import AIConfig

        difficulty = self._resolve_difficulty(agent, default_offset=2)
        config = AIConfig(difficulty=difficulty)
        heuristic_ai = HeuristicAI(player_number=0, config=config)

        # Wrapper to adapt select_move to get_best_move interface
        class HeuristicAgentWrapper:
            def __init__(self, ai):
                self._ai = ai

            def get_best_move(self, state, legal_moves):
                return self._ai.select_move(state)

        return HeuristicAgentWrapper(heuristic_ai)

    @staticmethod
    def _resolve_difficulty(agent: AIAgent, default_offset: int = 2) -> int:
        metadata = agent.metadata or {}
        if isinstance(metadata, dict) and "difficulty" in metadata:
            try:
                return int(metadata["difficulty"])
            except (ValueError, TypeError):
                pass
        return int(agent.search_depth + default_offset)

    _fallback_ai_counter: int = 0  # Class-level counter for unique seeds

    def _create_fallback_random_ai(self) -> Any:
        """Create a random AI instance for filling missing player slots.

        Used when tournaments register fewer agents than num_players in a game.
        The fallback AI simply picks random legal moves.

        Each fallback AI gets a unique seed to ensure varied behavior across
        multiple games (Jan 2026 bug fix).
        """
        import time

        from app.ai.random_ai import RandomAI
        from app.models import AIConfig

        # Unique seed per instance: counter + time-based entropy
        TournamentRunner._fallback_ai_counter += 1
        rng_seed = (
            TournamentRunner._fallback_ai_counter * 104729 +
            int(time.time() * 1000) % 1_000_000
        ) & 0xFFFFFFFF

        config = AIConfig(difficulty=1, rng_seed=rng_seed)
        random_ai = RandomAI(player_number=0, config=config)

        class RandomAgentWrapper:
            def __init__(self, ai):
                self._ai = ai

            def get_best_move(self, state, legal_moves):
                return self._ai.select_move(state)

        return RandomAgentWrapper(random_ai)

    def _execute_match_local(
        self,
        match: Match,
        agents: dict[str, AIAgent],
    ) -> MatchResult:
        """Execute a single match locally.

        This is the default local executor. Override with set_match_executor
        for distributed execution.
        """
        import time
        from contextlib import ExitStack

        from app.db.unified_recording import UnifiedGameRecorder, is_recording_enabled
        from app.game_engine import GameEngine
        from app.quality import compute_game_quality
        from app.rules.history_contract import derive_phase_from_move_type, phase_move_contract
        from app.training.initial_state import create_initial_state

        start_time = time.time()

        # Mark match as started
        self.scheduler.mark_match_started(match.match_id)

        # Initialize game
        state = create_initial_state(
            board_type=match.board_type,
            num_players=match.num_players,
        )

        # Create AI instances for each agent
        ai_instances = []
        for agent_id in match.agent_ids:
            agent = agents[agent_id]
            ai = self._create_ai_instance(agent, match.board_type, match.num_players)
            ai_instances.append(ai)

        # Fill missing player slots with random AI if needed
        # This handles cases where tournaments register fewer agents than num_players
        if len(ai_instances) < match.num_players:
            logger.warning(
                f"Match {match.match_id}: Only {len(ai_instances)} agents for "
                f"{match.num_players}-player game. Filling slots with random AI."
            )
            while len(ai_instances) < match.num_players:
                fallback_ai = self._create_fallback_random_ai()
                ai_instances.append(fallback_ai)

        recorded_move_types: list[str] = []
        recording_enabled = (
            self.recording_options is not None
            and self.recording_options.enabled
            and is_recording_enabled()
        )

        board_type_value = (
            match.board_type.value if hasattr(match.board_type, "value") else str(match.board_type)
        )

        with ExitStack() as stack:
            recorder = None
            if recording_enabled and self.recording_options is not None:
                recording_config = self.recording_options.build_recording_config(
                    board_type=board_type_value,
                    num_players=match.num_players,
                )
                recorder = stack.enter_context(
                    UnifiedGameRecorder(recording_config, state, game_id=match.match_id)
                )

            # Play the game
            # Note: Game uses 1-based player numbers but AI instances use 0-based indices
            move_count = 0
            while state.game_status == GameStatus.ACTIVE and move_count < self.max_moves:
                current_player = state.current_player
                # Convert 1-based player number to 0-based index
                ai_index = current_player - 1 if current_player > 0 else current_player
                if ai_index < 0 or ai_index >= len(ai_instances):
                    logger.error(f"Invalid player index: {ai_index} (player={current_player})")
                    break
                ai = ai_instances[ai_index]

                # Get valid moves for current player
                legal_moves = GameEngine.get_valid_moves(state, current_player)

                if not legal_moves:
                    # No interactive moves - check for bookkeeping phase requirements
                    requirement = GameEngine.get_phase_requirement(state, current_player)
                    if requirement is not None:
                        # Create appropriate bookkeeping move based on requirement type
                        move = self._create_bookkeeping_move(requirement, current_player)
                        if move is not None:
                            state_before = state
                            state = GameEngine.apply_move(state, move, trace_mode=True)
                            move_count += 1
                            recorded_move_types.append(move.type.value)
                            if recorder is not None:
                                recorder.add_move(
                                    move,
                                    state_after=state,
                                    state_before=state_before,
                                    available_moves_count=0,
                                )
                            continue
                    # No moves and no requirement - game is stuck or ended
                    break

                # Get AI move for interactive phase
                move = ai.get_best_move(state, legal_moves)
                if move is None:
                    # AI couldn't select a move, pick first legal move
                    move = legal_moves[0]

                # Get soft policy targets for training data
                move_probs = None
                if hasattr(ai, 'get_visit_distribution'):
                    try:
                        moves_dist, probs_dist = ai.get_visit_distribution()
                        if moves_dist and probs_dist:
                            move_probs = {}
                            for mv, prob in zip(moves_dist, probs_dist, strict=False):
                                if hasattr(mv, 'to') and mv.to is not None:
                                    move_key = f"{mv.to.x},{mv.to.y}"
                                    if hasattr(mv, 'from_pos') and mv.from_pos is not None:
                                        move_key = f"{mv.from_pos.x},{mv.from_pos.y}->{move_key}"
                                    move_probs[move_key] = float(prob)
                    except (AttributeError, ValueError, TypeError):
                        pass  # Silently ignore if visit distribution fails

                state_before = state
                state = GameEngine.apply_move(state, move, trace_mode=True)
                move_count += 1
                recorded_move_types.append(move.type.value)
                if recorder is not None:
                    recorder.add_move(
                        move,
                        state_after=state,
                        state_before=state_before,
                        available_moves_count=len(legal_moves),
                        move_probs=move_probs,
                    )

                # After each move, auto-process any bookkeeping requirements
                # This handles transitions through LINE_PROCESSING, TERRITORY_PROCESSING, etc.
                state, bookkeeping_moves = self._auto_process_bookkeeping(
                    state,
                    recorder=recorder,
                    recorded_move_types=recorded_move_types,
                )
                move_count += bookkeeping_moves

            # Determine rankings based on elimination order or ring counts
            rankings = self._compute_rankings(state, match.agent_ids)

            # Determine termination reason
            if state.game_status == GameStatus.COMPLETED:
                termination_reason = "completed"
            elif move_count >= self.max_moves:
                termination_reason = "max_moves"
            else:
                termination_reason = "no_moves"

            # Always determine a winner - use ranking[0] if no natural winner (2025-12-16 fix)
            # This ensures all games have a definite outcome for ELO updates
            if state.winner is not None:
                winner = rankings[0]
            elif rankings:
                # No natural winner but we have computed rankings via tiebreaker
                winner = rankings[0]
                logger.info(f"Game ended via {termination_reason}, winner by tiebreaker: {winner}")
            else:
                winner = None

            duration = time.time() - start_time

            # Build match metadata with agent information
            match_metadata: dict[str, Any] = {
                "board_type": match.board_type.value if hasattr(match.board_type, "value") else str(match.board_type),
                "num_players": match.num_players,
                "agents": {},
            }
            for agent_id in match.agent_ids:
                agent = agents[agent_id]
                agent_meta: dict[str, Any] = {
                    "agent_type": agent.agent_type.value if hasattr(agent.agent_type, "value") else str(agent.agent_type),
                    "version": agent.version,
                }
                if agent.model_path:
                    # Extract model metadata for neural agents
                    model_meta = extract_model_metadata(agent.model_path)
                    agent_meta["model_type"] = model_meta["model_type"]
                    agent_meta["model_class"] = model_meta["model_class"]
                    agent_meta["architecture_version"] = model_meta["architecture_version"]
                    agent_meta["training_board_type"] = model_meta["board_type"]
                    agent_meta["training_num_players"] = model_meta["num_players"]
                match_metadata["agents"][agent_id] = agent_meta

            if recorder is not None and self.recording_options is not None:
                unique_move_types = set(recorded_move_types)
                phase_labels = {
                    phase
                    for phase in (
                        derive_phase_from_move_type(mt) for mt in recorded_move_types
                    )
                    if phase
                }
                phase_count = len(phase_move_contract()) or 1
                phase_balance_score = min(1.0, len(phase_labels) / phase_count)
                diversity_score = min(1.0, len(unique_move_types) / 6.0) if unique_move_types else 0.0

                quality = compute_game_quality(
                    {
                        "game_id": match.match_id,
                        "move_count": move_count,
                        "board_type": board_type_value,
                        "num_players": match.num_players,
                        "winner": winner,
                        "termination_reason": termination_reason,
                        "source": self.recording_options.source,
                        "phase_balance_score": phase_balance_score,
                        "diversity_score": diversity_score,
                    }
                )

                base_metadata = {
                    "tournament_id": self.tournament_id or "default",
                    "match_id": match.match_id,
                    "round_number": match.round_number,
                    "worker_id": match.worker_id,
                    "game_length": move_count,
                    "termination_reason": termination_reason,
                    "winner": winner,
                    "agent_ids": match.agent_ids,
                    "agent_metadata": match_metadata["agents"],
                    "match_metadata": match.metadata,
                    "phase_balance_score": phase_balance_score,
                    "diversity_score": diversity_score,
                    "quality_score": quality.quality_score,
                    "quality_category": quality.category.value,
                    "training_weight": quality.training_weight,
                    "sync_priority": quality.sync_priority,
                }
                extra_metadata = dict(self.recording_options.extra_metadata or {})
                extra_metadata.update(base_metadata)
                recorder.finalize(state, extra_metadata=extra_metadata)

        return MatchResult(
            match_id=match.match_id,
            agent_ids=match.agent_ids,
            rankings=rankings,
            winner=winner,
            game_length=move_count,
            termination_reason=termination_reason,
            duration_seconds=duration,
            metadata=match_metadata,
        )

    def _compute_rankings(
        self,
        state: Any,  # GameState
        agent_ids: list[str],
    ) -> list[str]:
        """Compute player rankings from final game state.

        Note: Game uses 1-based player numbers but agent_ids is 0-indexed.
        """
        num_players = len(agent_ids)

        # If there's a winner, they're first
        # Note: state.winner is 1-based, convert to 0-based index
        if state.winner is not None:
            winner_idx = state.winner - 1 if state.winner > 0 else state.winner
            if 0 <= winner_idx < num_players:
                rankings = [agent_ids[winner_idx]]
                remaining = [
                    agent_ids[i] for i in range(num_players) if i != winner_idx
                ]
                # Sort remaining by elimination order (last eliminated = higher rank)
                # For now, just append them in reverse order
                rankings.extend(reversed(remaining))
                return rankings

        # For multiplayer without clear winner, rank by territory and rings
        player_scores = []
        for player_idx in range(num_players):
            # Score based on territory and rings (higher is better)
            # state.players is indexed 0, 1, ... but player.player_number is 1, 2, ...
            player = state.players[player_idx] if player_idx < len(state.players) else None
            if player:
                # More territory = better, fewer eliminated rings = better
                score = player.territory_spaces * 10 - player.eliminated_rings
            else:
                score = 0
            player_scores.append((player_idx, score))

        # Sort by score descending
        player_scores.sort(key=lambda x: x[1], reverse=True)
        return [agent_ids[player_idx] for player_idx, _ in player_scores]

    def _create_bookkeeping_move(self, requirement: Any, player: int) -> Any | None:
        """Create a bookkeeping move based on phase requirement.

        Args:
            requirement: PhaseRequirement from GameEngine.get_phase_requirement()
            player: The player number

        Returns:
            A Move object for the bookkeeping action, or None if unsupported
        """
        from datetime import datetime

        from app.game_engine import PhaseRequirementType
        from app.models import Move, MoveType

        req_type = requirement.type

        # Map requirement types to move types
        move_type_map = {
            PhaseRequirementType.NO_PLACEMENT_ACTION_REQUIRED: MoveType.NO_PLACEMENT_ACTION,
            PhaseRequirementType.NO_MOVEMENT_ACTION_REQUIRED: MoveType.NO_MOVEMENT_ACTION,
            PhaseRequirementType.NO_LINE_ACTION_REQUIRED: MoveType.NO_LINE_ACTION,
            PhaseRequirementType.NO_TERRITORY_ACTION_REQUIRED: MoveType.NO_TERRITORY_ACTION,
            PhaseRequirementType.FORCED_ELIMINATION_REQUIRED: MoveType.FORCED_ELIMINATION,
        }

        move_type = move_type_map.get(req_type)
        if move_type is None:
            logger.warning(f"Unsupported phase requirement type: {req_type}")
            return None

        return Move(
            id="bookkeeping",
            type=move_type,
            player=player,
            timestamp=datetime.now(),
        )

    def _auto_process_bookkeeping(
        self,
        state: Any,
        *,
        recorder: Any | None = None,
        recorded_move_types: list[str] | None = None,
    ) -> tuple[Any, int]:
        """Auto-process bookkeeping phases until an interactive phase is reached.

        This handles LINE_PROCESSING, TERRITORY_PROCESSING, and other non-interactive
        phases by generating and applying the required bookkeeping moves.

        Args:
            state: Current game state
            recorder: Optional recorder for canonical move history
            recorded_move_types: Optional list to append move types to

        Returns:
            Tuple of (updated_state, bookkeeping_moves_applied)
        """
        from app.game_engine import GameEngine

        max_bookkeeping_moves = 50  # Safety limit to prevent infinite loops
        bookkeeping_moves = 0

        for _ in range(max_bookkeeping_moves):
            if state.game_status != GameStatus.ACTIVE:
                break

            current_player = state.current_player

            # Check if there are interactive moves available
            legal_moves = GameEngine.get_valid_moves(state, current_player)
            if legal_moves:
                # Interactive moves exist - stop auto-processing
                break

            # No interactive moves - check for bookkeeping requirement
            requirement = GameEngine.get_phase_requirement(state, current_player)
            if requirement is None:
                # No requirement and no moves - game may be stuck
                break

            # Create and apply bookkeeping move
            move = self._create_bookkeeping_move(requirement, current_player)
            if move is None:
                break

            state_before = state
            state = GameEngine.apply_move(state, move, trace_mode=True)
            bookkeeping_moves += 1
            if recorded_move_types is not None:
                recorded_move_types.append(move.type.value)
            if recorder is not None:
                recorder.add_move(
                    move,
                    state_after=state,
                    state_before=state_before,
                    available_moves_count=0,
                )

        return state, bookkeeping_moves

    def get_leaderboard(self) -> list[tuple[str, float, dict]]:
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
