"""Massive Tournament Orchestrator - Coordinate cluster-wide model evaluation.

This module orchestrates large-scale tournaments across the entire cluster,
enabling Elo evaluation of thousands of models against varied opponents.

Usage:
    from app.coordination.massive_tournament_orchestrator import (
        MassiveTournamentOrchestrator,
        MassiveTournamentConfig,
    )
    from app.utils.model_deduplicator import ModelDeduplicator

    deduplicator = ModelDeduplicator()
    unique_models = await deduplicator.scan_directory(Path("/Volumes/RingRift-Data"))

    config = MassiveTournamentConfig(games_per_opponent=25)
    orchestrator = MassiveTournamentOrchestrator(deduplicator, config)

    results = await orchestrator.run_full_tournament([Path("/Volumes/RingRift-Data")])

January 2, 2026: Created for massive Elo evaluation.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.utils.model_deduplicator import ModelDeduplicator, UniqueModel

logger = logging.getLogger(__name__)

# Default opponents for tournament
DEFAULT_OPPONENTS = [
    "random",      # Random move baseline
    "heuristic",   # Heuristic AI (depth 5)
    "canonical",   # Canonical model for that config
    "mcts_25",     # MCTS with 25 simulations
]


@dataclass
class TournamentWorkItem:
    """A single work item for tournament evaluation."""

    work_id: str
    model: UniqueModel
    opponent_type: str
    games: int
    board_type: str
    num_players: int
    status: str = "pending"  # pending, running, completed, failed
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games_played: int = 0
    error: str | None = None
    worker_id: str | None = None
    started_at: float | None = None
    completed_at: float | None = None


@dataclass
class TournamentProgress:
    """Progress tracking for a config tournament."""

    config_key: str
    total_models: int
    total_work_items: int
    completed_work_items: int = 0
    total_games: int = 0
    games_played: int = 0
    started_at: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)

    @property
    def percent_complete(self) -> float:
        """Get completion percentage."""
        if self.total_work_items == 0:
            return 0.0
        return (self.completed_work_items / self.total_work_items) * 100

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.started_at

    @property
    def games_per_hour(self) -> float:
        """Get games played per hour."""
        elapsed_hours = self.elapsed_seconds / 3600
        if elapsed_hours < 0.001:
            return 0.0
        return self.games_played / elapsed_hours


@dataclass
class ConfigTournamentResult:
    """Result of a config tournament."""

    config_key: str
    total_models: int
    total_games: int
    elapsed_seconds: float
    model_results: dict[str, dict[str, Any]]  # sha256 -> {wins, losses, elo, etc.}
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config_key": self.config_key,
            "total_models": self.total_models,
            "total_games": self.total_games,
            "elapsed_seconds": self.elapsed_seconds,
            "games_per_hour": (self.total_games / self.elapsed_seconds * 3600)
            if self.elapsed_seconds > 0
            else 0,
            "error_count": len(self.errors),
        }


@dataclass
class TournamentResults:
    """Aggregate results from full tournament."""

    tournament_id: str
    started_at: float
    completed_at: float | None = None
    total_unique_models: int = 0
    total_games: int = 0
    config_results: dict[str, ConfigTournamentResult] = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed

    @property
    def elapsed_seconds(self) -> float:
        """Get total elapsed time."""
        end = self.completed_at or time.time()
        return end - self.started_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to summary dictionary."""
        return {
            "tournament_id": self.tournament_id,
            "status": self.status,
            "total_unique_models": self.total_unique_models,
            "total_games": self.total_games,
            "elapsed_seconds": self.elapsed_seconds,
            "elapsed_hours": self.elapsed_seconds / 3600,
            "games_per_hour": (self.total_games / self.elapsed_seconds * 3600)
            if self.elapsed_seconds > 0
            else 0,
            "configs_completed": len(
                [r for r in self.config_results.values() if r.total_games > 0]
            ),
            "total_configs": len(self.config_results),
        }


@dataclass
class MassiveTournamentConfig:
    """Configuration for massive model tournament."""

    games_per_opponent: int = 25  # 25 games x 4 opponents = 100 games
    opponents: list[str] = field(default_factory=lambda: DEFAULT_OPPONENTS.copy())
    parallel_configs: int = 12  # Evaluate all 12 configs in parallel
    games_batch_size: int = 64  # Games per P2P job dispatch
    record_games: bool = True  # Save for training
    recording_db_prefix: str = "tournament_massive"
    min_elo_games: int = 100  # Target 100+ games for confidence
    game_timeout_seconds: int = 300  # 5 minutes per game
    work_item_timeout_seconds: int = 1800  # 30 minutes per work item
    checkpoint_interval: int = 100  # Checkpoint every N completed work items
    resume_from_checkpoint: str | None = None  # Checkpoint ID to resume from

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.games_per_opponent < 1:
            raise ValueError("games_per_opponent must be >= 1")
        if len(self.opponents) < 1:
            raise ValueError("At least one opponent required")


class MassiveTournamentOrchestrator(HandlerBase):
    """Orchestrate massive cluster-wide tournament.

    This orchestrator coordinates:
    1. Model deduplication and scanning
    2. Work item generation for (model, opponent) pairs
    3. Parallel dispatch across cluster nodes
    4. Progress tracking and checkpointing
    5. Result aggregation and Elo updates
    """

    _instance: MassiveTournamentOrchestrator | None = None

    def __init__(
        self,
        deduplicator: ModelDeduplicator | None = None,
        config: MassiveTournamentConfig | None = None,
    ):
        """Initialize the orchestrator."""
        super().__init__(
            name="massive_tournament_orchestrator",
            cycle_interval=60.0,  # Progress check every minute
        )
        self._deduplicator = deduplicator or ModelDeduplicator()
        self._config = config or MassiveTournamentConfig()
        self._progress: dict[str, TournamentProgress] = {}
        self._work_items: dict[str, TournamentWorkItem] = {}
        self._current_tournament: TournamentResults | None = None
        self._checkpoint_db: Path = Path("data/tournament_checkpoint.db")

    @classmethod
    def get_instance(cls) -> MassiveTournamentOrchestrator:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    async def _run_cycle(self) -> None:
        """Check tournament progress and emit events."""
        if not self._current_tournament:
            return

        # Update progress
        total_games = 0
        games_played = 0
        for progress in self._progress.values():
            total_games += progress.total_games
            games_played += progress.games_played

        # Emit progress event
        try:
            from app.coordination.data_events import DataEventType
            from app.coordination.event_router import emit_event

            emit_event(
                DataEventType.MASSIVE_TOURNAMENT_PROGRESS
                if hasattr(DataEventType, "MASSIVE_TOURNAMENT_PROGRESS")
                else "massive_tournament_progress",
                {
                    "tournament_id": self._current_tournament.tournament_id,
                    "total_games": total_games,
                    "games_played": games_played,
                    "percent_complete": (games_played / total_games * 100)
                    if total_games > 0
                    else 0,
                    "elapsed_hours": self._current_tournament.elapsed_seconds / 3600,
                },
            )
        except ImportError:
            pass

    async def run_full_tournament(
        self,
        model_sources: list[Path],
    ) -> TournamentResults:
        """Run full tournament across all configs.

        Args:
            model_sources: Directories to scan for models

        Returns:
            TournamentResults with aggregated results
        """
        tournament_id = str(uuid.uuid4())[:8]
        self._current_tournament = TournamentResults(
            tournament_id=tournament_id,
            started_at=time.time(),
            status="running",
        )

        logger.info(f"Starting tournament {tournament_id}")

        # Emit start event
        self._emit_tournament_event("MASSIVE_TOURNAMENT_STARTED", {
            "tournament_id": tournament_id,
            "model_sources": [str(p) for p in model_sources],
            "config": {
                "games_per_opponent": self._config.games_per_opponent,
                "opponents": self._config.opponents,
                "record_games": self._config.record_games,
            },
        })

        try:
            # Phase 1: Scan and deduplicate models
            logger.info("Phase 1: Scanning for unique models...")
            unique_models = []
            for source in model_sources:
                models = await self._deduplicator.scan_directory(source)
                unique_models.extend(models)

            # Deduplicate across sources (in case same model in multiple sources)
            seen_sha256 = set()
            final_models = []
            for model in unique_models:
                if model.sha256 not in seen_sha256:
                    seen_sha256.add(model.sha256)
                    final_models.append(model)

            self._current_tournament.total_unique_models = len(final_models)
            logger.info(f"Found {len(final_models)} unique models across all sources")

            # Phase 2: Group by config
            models_by_config = self._deduplicator.group_by_config(final_models)
            logger.info(f"Models grouped into {len(models_by_config)} configs")

            # Phase 3: Launch parallel tournaments per config
            logger.info("Phase 3: Running config tournaments in parallel...")
            tasks = []
            for config_key, models in models_by_config.items():
                logger.info(f"  {config_key}: {len(models)} models")
                task = self._run_config_tournament(config_key, models)
                tasks.append(task)

            # Run all configs in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Phase 4: Aggregate results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Config tournament failed: {result}")
                    continue
                if isinstance(result, ConfigTournamentResult):
                    self._current_tournament.config_results[result.config_key] = result
                    self._current_tournament.total_games += result.total_games

            self._current_tournament.completed_at = time.time()
            self._current_tournament.status = "completed"

            logger.info(
                f"Tournament {tournament_id} completed: "
                f"{self._current_tournament.total_games} games "
                f"in {self._current_tournament.elapsed_seconds / 3600:.1f} hours"
            )

            # Emit completion event
            self._emit_tournament_event("MASSIVE_TOURNAMENT_COMPLETED", {
                "tournament_id": tournament_id,
                **self._current_tournament.to_dict(),
            })

        except Exception as e:
            self._current_tournament.status = "failed"
            logger.error(f"Tournament {tournament_id} failed: {e}")
            raise

        return self._current_tournament

    async def _run_config_tournament(
        self,
        config_key: str,
        models: list[UniqueModel],
    ) -> ConfigTournamentResult:
        """Run tournament for a single config.

        Args:
            config_key: Config key (e.g., "hex8_2p")
            models: List of unique models for this config

        Returns:
            ConfigTournamentResult with per-model results
        """
        from app.coordination.event_utils import parse_config_key

        parsed = parse_config_key(config_key)
        board_type = parsed.board_type
        num_players = parsed.num_players

        # Initialize progress tracking
        total_work_items = len(models) * len(self._config.opponents)
        progress = TournamentProgress(
            config_key=config_key,
            total_models=len(models),
            total_work_items=total_work_items,
            total_games=total_work_items * self._config.games_per_opponent,
        )
        self._progress[config_key] = progress

        logger.info(
            f"[{config_key}] Starting tournament: "
            f"{len(models)} models x {len(self._config.opponents)} opponents "
            f"x {self._config.games_per_opponent} games = "
            f"{progress.total_games} total games"
        )

        # Create work items
        work_items = []
        for model in models:
            for opponent in self._config.opponents:
                work_id = f"{config_key}_{model.sha256[:8]}_{opponent}"
                work_item = TournamentWorkItem(
                    work_id=work_id,
                    model=model,
                    opponent_type=opponent,
                    games=self._config.games_per_opponent,
                    board_type=board_type,
                    num_players=num_players,
                )
                work_items.append(work_item)
                self._work_items[work_id] = work_item

        # Dispatch work items
        model_results: dict[str, dict[str, Any]] = {}
        start_time = time.time()

        # Process in batches
        batch_size = self._config.games_batch_size
        for batch_start in range(0, len(work_items), batch_size):
            batch_end = min(batch_start + batch_size, len(work_items))
            batch = work_items[batch_start:batch_end]

            # Dispatch batch to workers
            batch_results = await self._dispatch_work_batch(batch)

            # Process results
            for work_item, result in batch_results:
                sha256 = work_item.model.sha256
                if sha256 not in model_results:
                    model_results[sha256] = {
                        "model_path": str(work_item.model.canonical_path),
                        "model_family": work_item.model.model_family,
                        "total_games": 0,
                        "total_wins": 0,
                        "total_losses": 0,
                        "total_draws": 0,
                        "opponents": {},
                    }

                if result:
                    model_results[sha256]["total_games"] += result.get("games_played", 0)
                    model_results[sha256]["total_wins"] += result.get("wins", 0)
                    model_results[sha256]["total_losses"] += result.get("losses", 0)
                    model_results[sha256]["total_draws"] += result.get("draws", 0)
                    model_results[sha256]["opponents"][work_item.opponent_type] = result

                progress.completed_work_items += 1
                progress.games_played += work_item.games_played
                progress.last_update = time.time()

            # Log progress
            if progress.completed_work_items % 10 == 0:
                logger.info(
                    f"[{config_key}] Progress: "
                    f"{progress.completed_work_items}/{progress.total_work_items} "
                    f"({progress.percent_complete:.1f}%) - "
                    f"{progress.games_per_hour:.0f} games/hour"
                )

        elapsed = time.time() - start_time

        # Emit config completion event
        self._emit_tournament_event("MASSIVE_TOURNAMENT_CONFIG_COMPLETE", {
            "config_key": config_key,
            "total_models": len(models),
            "total_games": progress.games_played,
            "elapsed_seconds": elapsed,
        })

        return ConfigTournamentResult(
            config_key=config_key,
            total_models=len(models),
            total_games=progress.games_played,
            elapsed_seconds=elapsed,
            model_results=model_results,
        )

    async def _dispatch_work_batch(
        self,
        work_items: list[TournamentWorkItem],
    ) -> list[tuple[TournamentWorkItem, dict[str, Any] | None]]:
        """Dispatch a batch of work items to workers.

        Returns:
            List of (work_item, result) tuples
        """
        results: list[tuple[TournamentWorkItem, dict[str, Any] | None]] = []

        # Try to use P2P cluster for distributed execution
        try:
            distributed_results = await self._dispatch_distributed(work_items)
            results.extend(distributed_results)
        except Exception as e:
            logger.warning(f"Distributed dispatch failed: {e}, falling back to local")
            # Fall back to local execution
            for work_item in work_items:
                result = await self._run_work_item_local(work_item)
                results.append((work_item, result))

        return results

    async def _dispatch_distributed(
        self,
        work_items: list[TournamentWorkItem],
    ) -> list[tuple[TournamentWorkItem, dict[str, Any] | None]]:
        """Dispatch work items to P2P cluster workers."""
        # Try to get P2P status and available workers
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                # Get cluster status
                async with session.get(
                    "http://localhost:8770/status",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError("P2P not available")
                    status = await resp.json()

                alive_peers = status.get("alive_peers", 0)
                if alive_peers < 1:
                    raise RuntimeError("No alive peers")

                logger.debug(f"Dispatching to {alive_peers} cluster workers")

                # For now, fall back to local execution
                # TODO: Implement full P2P job dispatch
                results = []
                for work_item in work_items:
                    result = await self._run_work_item_local(work_item)
                    results.append((work_item, result))
                return results

        except Exception as e:
            raise RuntimeError(f"Distributed dispatch failed: {e}") from e

    async def _run_work_item_local(
        self,
        work_item: TournamentWorkItem,
    ) -> dict[str, Any] | None:
        """Run a work item locally."""
        work_item.status = "running"
        work_item.started_at = time.time()

        try:
            from app.training.game_gauntlet import (
                BaselineOpponent,
                run_baseline_gauntlet,
            )

            # Map opponent type to BaselineOpponent
            opponent_map = {
                "random": BaselineOpponent.RANDOM,
                "heuristic": BaselineOpponent.HEURISTIC,
                "mcts_25": BaselineOpponent.MCTS,
                "canonical": BaselineOpponent.NEURAL_NET,
            }

            opponent = opponent_map.get(work_item.opponent_type)
            if opponent is None:
                logger.warning(f"Unknown opponent type: {work_item.opponent_type}")
                work_item.status = "failed"
                work_item.error = f"Unknown opponent: {work_item.opponent_type}"
                return None

            # Import board type
            from app.rules.board import BoardType

            board_type_map = {
                "hex8": BoardType.HEX8,
                "square8": BoardType.SQUARE8,
                "square19": BoardType.SQUARE19,
                "hexagonal": BoardType.HEXAGONAL,
            }
            board_type = board_type_map.get(work_item.board_type)
            if board_type is None:
                logger.warning(f"Unknown board type: {work_item.board_type}")
                work_item.status = "failed"
                work_item.error = f"Unknown board: {work_item.board_type}"
                return None

            # Run gauntlet
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: run_baseline_gauntlet(
                    model_path=str(work_item.model.canonical_path),
                    board_type=board_type,
                    num_players=work_item.num_players,
                    opponents=[opponent],
                    games_per_opponent=work_item.games,
                    record_games=self._config.record_games,
                ),
            )

            # Extract result for this opponent
            opponent_key = work_item.opponent_type
            if opponent_key in results:
                result = results[opponent_key]
                work_item.wins = result.get("wins", 0)
                work_item.losses = result.get("losses", 0)
                work_item.draws = result.get("draws", 0)
                work_item.games_played = result.get("games_played", 0)
                work_item.status = "completed"
                work_item.completed_at = time.time()

                # Emit per-model evaluation event
                self._emit_evaluation_completed(work_item)

                return result

            work_item.status = "failed"
            work_item.error = "No result for opponent"
            return None

        except Exception as e:
            logger.error(f"Work item {work_item.work_id} failed: {e}")
            work_item.status = "failed"
            work_item.error = str(e)
            work_item.completed_at = time.time()
            return None

    def _emit_tournament_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit a tournament event."""
        try:
            from app.coordination.event_router import emit_event

            emit_event(event_type, data)
        except Exception as e:
            logger.debug(f"Could not emit event {event_type}: {e}")

    def _emit_evaluation_completed(self, work_item: TournamentWorkItem) -> None:
        """Emit EVALUATION_COMPLETED event for curriculum integration."""
        try:
            from app.coordination.data_events import DataEventType
            from app.coordination.event_router import emit_event

            config_key = f"{work_item.board_type}_{work_item.num_players}p"
            win_rate = (
                work_item.wins / work_item.games_played
                if work_item.games_played > 0
                else 0.5
            )

            emit_event(
                DataEventType.EVALUATION_COMPLETED,
                {
                    "config_key": config_key,
                    "model_path": str(work_item.model.canonical_path),
                    "model_sha256": work_item.model.sha256,
                    "opponent_type": work_item.opponent_type,
                    "games_played": work_item.games_played,
                    "wins": work_item.wins,
                    "losses": work_item.losses,
                    "draws": work_item.draws,
                    "win_rate": win_rate,
                    "source": "massive_tournament",
                },
            )
        except Exception as e:
            logger.debug(f"Could not emit EVALUATION_COMPLETED: {e}")

    def health_check(self) -> HealthCheckResult:
        """Return health status."""
        status = "healthy" if self._current_tournament is None else "busy"

        details = {
            "tournament_active": self._current_tournament is not None,
        }

        if self._current_tournament:
            details["tournament_id"] = self._current_tournament.tournament_id
            details["status"] = self._current_tournament.status
            details["total_games"] = self._current_tournament.total_games
            details["elapsed_hours"] = self._current_tournament.elapsed_seconds / 3600

        return HealthCheckResult(
            name=self._name,
            status=status,
            details=details,
        )


# Module-level helpers for easy access
def get_tournament_orchestrator() -> MassiveTournamentOrchestrator:
    """Get singleton tournament orchestrator."""
    return MassiveTournamentOrchestrator.get_instance()
