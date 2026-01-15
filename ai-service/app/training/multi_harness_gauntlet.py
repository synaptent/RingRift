"""Multi-Harness Gauntlet for evaluating models under all compatible AI algorithms.

This module automatically evaluates a trained model under multiple AI harnesses
(algorithms) and tracks the Elo ratings for each combination. This enables:

1. Finding the best (model, harness) combination for deployment
2. Tracking architecture performance across different usage patterns
3. Generating training data from diverse algorithm perspectives

Model Types:
    - nn: Full neural network (v2-v6)
    - nnue: NNUE evaluation network (2-player only)
    - nnue_mp: Multi-player NNUE (3-4 player)

Harness Types:
    - policy_only: Direct policy sampling (fast, lower quality)
    - mcts: Monte Carlo Tree Search
    - gumbel_mcts: Gumbel MCTS (best quality)
    - descent: Gradient descent search
    - minimax: Alpha-beta minimax (NNUE)
    - maxn: Max-N search (multi-player)
    - brs: Best-Reply Search (multi-player)

December 2025: Updated to integrate with Phase 1 harness abstraction layer.
- Uses app.ai.harness for unified harness creation
- Integrates with NNUE baseline opponents for evaluation
- Captures visit distributions for soft policy targets

Usage:
    from app.training.multi_harness_gauntlet import MultiHarnessGauntlet

    gauntlet = MultiHarnessGauntlet()
    results = await gauntlet.evaluate_model(
        model_path="models/canonical_hex8_4p.pth",
        model_type="nn",
        board_type="hex8",
        num_players=4,
    )
    for harness, rating in results.items():
        print(f"  {harness}: {rating.elo:.0f}")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

# December 30, 2025: Game count for graduated thresholds
from app.utils.game_discovery import get_game_counts_summary

if TYPE_CHECKING:
    from app.models import BoardType


# Import canonical types from harness_registry (Jan 2026 consolidation)
# HarnessType and ModelType are now imported from base_harness.py
# Previously this file had local enum definitions that have been removed
from app.ai.harness.base_harness import HarnessType, ModelType
from app.ai.harness.harness_registry import (
    get_harness_matrix,
    get_harness_player_range,
    get_harnesses_for_model_and_players,
    is_harness_valid_for_player_count,
    HARNESS_PLAYER_RESTRICTIONS,
)

# Backward compatibility: HARNESS_MATRIX is now computed from canonical source
# Use get_harness_matrix() for current values
HARNESS_MATRIX: dict[ModelType, list[HarnessType]] = get_harness_matrix()


@dataclass
class EloRating:
    """Elo rating result from gauntlet evaluation."""

    elo: float
    games_played: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    confidence_interval: tuple[float, float] = (0.0, 0.0)

    @property
    def display(self) -> str:
        """Human-readable display string."""
        return (
            f"Elo {self.elo:.0f} (±{self.confidence_interval[1] - self.elo:.0f}) "
            f"[{self.wins}W/{self.losses}L/{self.draws}D] "
            f"({self.win_rate:.1%} win rate)"
        )


@dataclass
class HarnessConfig:
    """Configuration for a specific harness evaluation."""

    harness_type: HarnessType
    games_per_baseline: int = 30
    baselines: list[str] = field(default_factory=lambda: ["random", "heuristic"])
    timeout_seconds: float = 300.0
    parallel_games: int = 4

    # Harness-specific settings
    mcts_budget: int | None = None  # Budget for MCTS variants
    search_depth: int | None = None  # Depth for minimax/maxn/brs
    temperature: float | None = None  # Temperature for policy sampling


@dataclass
class MultiHarnessResult:
    """Result of multi-harness evaluation."""

    model_path: str | None = None
    model_type: ModelType | None = None
    board_type: str | None = None
    num_players: int = 2
    harness_results: dict[HarnessType, Any] = field(default_factory=dict)
    best_harness: HarnessType | str | None = None
    best_elo: float = 0.0
    total_games: int = 0
    evaluation_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Handle harness results - may be EloRating or GauntletResult objects
        harness_data = {}
        for h, r in self.harness_results.items():
            h_key = h.value if hasattr(h, "value") else str(h)
            if hasattr(r, "elo"):
                harness_data[h_key] = {
                    "elo": r.elo,
                    "games_played": getattr(r, "games_played", 0),
                    "wins": getattr(r, "wins", 0),
                    "losses": getattr(r, "losses", 0),
                    "draws": getattr(r, "draws", 0),
                    "win_rate": getattr(r, "win_rate", 0.0),
                }
            else:
                harness_data[h_key] = r

        # Handle best_harness which may be str or enum
        best_harness_str = None
        if self.best_harness is not None:
            best_harness_str = (
                self.best_harness.value
                if hasattr(self.best_harness, "value")
                else str(self.best_harness)
            )

        return {
            "model_path": self.model_path,
            "model_type": self.model_type.value if self.model_type else None,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "harness_results": harness_data,
            "best_harness": best_harness_str,
            "best_elo": self.best_elo,
            "total_games": self.total_games,
            "evaluation_time_seconds": self.evaluation_time_seconds,
        }


class MultiHarnessGauntlet:
    """Evaluates a model under all compatible harnesses.

    This enables:
    1. Finding the best (model, harness) combination for deployment
    2. Tracking architecture performance across usage patterns
    3. Generating training data from diverse algorithm perspectives
    """

    def __init__(
        self,
        default_games_per_baseline: int = 30,
        default_baselines: list[str] | None = None,
        parallel_evaluations: int = 1,  # Sequential by default for stability
    ):
        """Initialize the multi-harness gauntlet.

        Args:
            default_games_per_baseline: Default games per baseline opponent
            default_baselines: Default baseline opponents to test against
            parallel_evaluations: Number of harness evaluations to run in parallel
        """
        self.default_games_per_baseline = default_games_per_baseline
        self.default_baselines = default_baselines or ["random", "heuristic"]
        self.parallel_evaluations = parallel_evaluations

    def get_compatible_harnesses(
        self,
        model_type: ModelType | str,
        num_players: int,
    ) -> list[HarnessType]:
        """Get list of harnesses compatible with model type and player count.

        Args:
            model_type: Model architecture type
            num_players: Number of players in the game

        Returns:
            List of compatible harness types
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        compatible = []
        # Use consolidated function from harness_registry (Dec 2025)
        for harness in HARNESS_MATRIX.get(model_type, []):
            if is_harness_valid_for_player_count(harness, num_players):
                compatible.append(harness)

        return compatible

    def detect_model_type(self, model_path: str | Path) -> ModelType:
        """Detect model type from checkpoint file.

        Args:
            model_path: Path to model checkpoint

        Returns:
            Detected ModelType
        """
        path = Path(model_path)
        name = path.stem.lower()

        # Check for explicit type markers in filename
        if "nnue_mp" in name or "multiplayer_nnue" in name:
            return ModelType.NNUE
        if "nnue" in name:
            return ModelType.NNUE

        # Check checkpoint metadata
        try:
            from app.utils.torch_utils import safe_load_checkpoint

            checkpoint = safe_load_checkpoint(str(path), map_location="cpu")
            if isinstance(checkpoint, dict):
                if checkpoint.get("model_type") == "nnue_mp":
                    return ModelType.NNUE
                if checkpoint.get("model_type") == "nnue":
                    return ModelType.NNUE
                if "num_players" in checkpoint and checkpoint["num_players"] >= 3:
                    # Multi-player NNUE if num_players stored
                    if "nnue" in name:
                        return ModelType.NNUE
        except (FileNotFoundError, ValueError, RuntimeError):
            pass

        # Default to full NN
        return ModelType.NEURAL_NET

    async def evaluate_model(
        self,
        model_path: str | Path,
        model_type: ModelType | str | None = None,
        board_type: str = "square8",
        num_players: int = 2,
        harnesses: list[HarnessType | str] | None = None,
        harness_configs: dict[HarnessType, HarnessConfig] | None = None,
        save_games: bool = True,
    ) -> MultiHarnessResult:
        """Evaluate a model under all compatible harnesses.

        Args:
            model_path: Path to model checkpoint
            model_type: Model type (auto-detected if None)
            board_type: Board type string (e.g., "hex8", "square8")
            num_players: Number of players
            harnesses: Specific harnesses to evaluate (None = all compatible)
            harness_configs: Override configurations per harness
            save_games: Whether to save games for training (default True)

        Returns:
            MultiHarnessResult with Elo ratings for each harness
        """
        import time

        start_time = time.time()
        model_path = Path(model_path)

        # Auto-detect model type if not provided
        if model_type is None:
            model_type = self.detect_model_type(model_path)
        elif isinstance(model_type, str):
            model_type = ModelType(model_type)

        # Get compatible harnesses
        if harnesses is None:
            harnesses = self.get_compatible_harnesses(model_type, num_players)
        else:
            harnesses = [
                HarnessType(h) if isinstance(h, str) else h for h in harnesses
            ]

        logger.info(
            f"MultiHarnessGauntlet: Evaluating {model_path.name} "
            f"({model_type.value}) under {len(harnesses)} harnesses: "
            f"{[h.value for h in harnesses]}"
        )

        # Evaluate under each harness
        # January 5, 2026: Use parallel evaluation if parallel_evaluations > 1
        harness_results: dict[HarnessType, EloRating] = {}
        total_games = 0

        if self.parallel_evaluations > 1 and len(harnesses) > 1:
            # Parallel evaluation using asyncio.gather with semaphore for concurrency limit
            logger.info(
                f"  Running {len(harnesses)} harnesses in parallel "
                f"(max {self.parallel_evaluations} concurrent)"
            )
            semaphore = asyncio.Semaphore(self.parallel_evaluations)

            async def evaluate_with_semaphore(
                harness: HarnessType,
            ) -> tuple[HarnessType, EloRating | None, str | None]:
                """Evaluate a single harness with concurrency limiting."""
                config = (harness_configs or {}).get(
                    harness, self._default_harness_config(harness)
                )
                async with semaphore:
                    try:
                        rating = await self._evaluate_single_harness(
                            model_path=model_path,
                            model_type=model_type,
                            board_type=board_type,
                            num_players=num_players,
                            harness=harness,
                            config=config,
                            save_games=save_games,
                        )
                        return (harness, rating, None)
                    except (RuntimeError, ValueError, TimeoutError) as e:
                        return (harness, None, str(e))

            # Launch all evaluations in parallel
            tasks = [evaluate_with_semaphore(h) for h in harnesses]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"  Harness evaluation error: {result}")
                    continue
                harness, rating, error = result
                if rating is not None:
                    harness_results[harness] = rating
                    total_games += rating.games_played
                    logger.info(f"  {harness.value}: {rating.display}")
                else:
                    logger.warning(f"  {harness.value}: FAILED - {error}")
        else:
            # Sequential evaluation (original behavior)
            for harness in harnesses:
                config = (harness_configs or {}).get(
                    harness, self._default_harness_config(harness)
                )

                try:
                    rating = await self._evaluate_single_harness(
                        model_path=model_path,
                        model_type=model_type,
                        board_type=board_type,
                        num_players=num_players,
                        harness=harness,
                        config=config,
                        save_games=save_games,
                    )
                    harness_results[harness] = rating
                    total_games += rating.games_played
                    logger.info(f"  {harness.value}: {rating.display}")
                except (RuntimeError, ValueError, TimeoutError) as e:
                    logger.warning(f"  {harness.value}: FAILED - {e}")

        # Find best harness
        best_harness = None
        best_elo = None
        if harness_results:
            best_harness = max(harness_results, key=lambda h: harness_results[h].elo)
            best_elo = harness_results[best_harness].elo

        evaluation_time = time.time() - start_time

        return MultiHarnessResult(
            model_path=str(model_path),
            model_type=model_type,
            board_type=board_type,
            num_players=num_players,
            harness_results=harness_results,
            best_harness=best_harness,
            best_elo=best_elo,
            total_games=total_games,
            evaluation_time_seconds=evaluation_time,
        )

    async def _evaluate_single_harness(
        self,
        model_path: Path,
        model_type: ModelType,
        board_type: str,
        num_players: int,
        harness: HarnessType,
        config: HarnessConfig,
        save_games: bool = True,
    ) -> EloRating:
        """Evaluate model under a single harness.

        Args:
            model_path: Path to model checkpoint
            model_type: Model architecture type
            board_type: Board type string
            num_players: Number of players
            harness: Harness to evaluate under
            config: Harness configuration
            save_games: Whether to save games for training (default True)

        Returns:
            EloRating for this harness
        """
        from app.training.game_gauntlet import (
            BaselineOpponent,
            GauntletGameResult,
            run_baseline_gauntlet,
            _create_gauntlet_recording_config,
        )
        from app.models import AIType, BoardType as BT

        # Convert board type string to enum
        board_type_enum = BT(board_type)

        # Map harness to AIType for gauntlet
        # Jan 2026: Removed HarnessType.MCTS (use GUMBEL_MCTS or GPU_GUMBEL)
        harness_to_ai_type = {
            HarnessType.POLICY_ONLY: AIType.POLICY_ONLY,
            HarnessType.GUMBEL_MCTS: AIType.GUMBEL_MCTS,
            HarnessType.GPU_GUMBEL: AIType.GUMBEL_MCTS,  # GPU batch uses same AI type
            HarnessType.DESCENT: AIType.DESCENT,
            HarnessType.MINIMAX: AIType.MINIMAX,
            HarnessType.MAXN: AIType.MAXN,
            HarnessType.BRS: AIType.BRS,
        }

        ai_type = harness_to_ai_type.get(harness)
        if ai_type is None:
            raise ValueError(f"Unknown harness type: {harness}")

        # Map baseline strings to enum
        # Jan 13, 2026: Complete mapping for all baselines including NNUE harness diversity
        baseline_map = {
            "random": BaselineOpponent.RANDOM,
            "heuristic": BaselineOpponent.HEURISTIC,
            "heuristic_strong": BaselineOpponent.HEURISTIC_STRONG,
            "mcts_light": BaselineOpponent.MCTS_LIGHT,
            "mcts_medium": BaselineOpponent.MCTS_MEDIUM,
            "mcts_strong": BaselineOpponent.MCTS_STRONG,
            "mcts_master": BaselineOpponent.MCTS_MASTER,
            "mcts_grandmaster": BaselineOpponent.MCTS_GRANDMASTER,
            "gumbel_b64": BaselineOpponent.GUMBEL_B64,
            "gumbel_b200": BaselineOpponent.GUMBEL_B200,
            "gumbel_nnue": BaselineOpponent.GUMBEL_NNUE,
            "policy_only_nn": BaselineOpponent.POLICY_ONLY_NN,
            "policy_only_nnue": BaselineOpponent.POLICY_ONLY_NNUE,
            "descent_nn": BaselineOpponent.DESCENT_NN,
            "descent_nnue": BaselineOpponent.DESCENT_NNUE,
            # NNUE baselines for harness diversity
            "nnue_minimax_d4": BaselineOpponent.NNUE_MINIMAX_D4,
            "nnue_maxn_d3": BaselineOpponent.NNUE_MAXN_D3,
            "nnue_brs_d3": BaselineOpponent.NNUE_BRS_D3,
        }
        baselines = [baseline_map[b] for b in config.baselines if b in baseline_map]

        # Dec 30, 2025: Get game count for graduated thresholds
        config_key = f"{board_type}_{num_players}p"
        try:
            game_counts = get_game_counts_summary()
            game_count = game_counts.get(config_key, 0)
        except (OSError, RuntimeError):
            game_count = None  # Will use fallback thresholds

        # Run gauntlet with this harness (in thread pool to not block)
        # January 9, 2026 (Sprint 17.9): Pass harness_type for composite participant IDs
        # and save_games_for_training for capturing full training data (move_probs, search_stats)
        # Jan 13, 2026: Create recording config to capture gauntlet games in database
        recording_config = _create_gauntlet_recording_config(
            board_type=board_type_enum,
            num_players=num_players,
            source=f"gauntlet_{harness.value}",
        )

        def run_gauntlet() -> dict[str, GauntletGameResult]:
            return run_baseline_gauntlet(
                model_path=str(model_path),
                board_type=board_type_enum,
                num_players=num_players,
                opponents=baselines,
                games_per_opponent=config.games_per_baseline,
                model_type=ai_type.value if ai_type else "cnn",  # Convert AIType to string
                parallel_games=config.parallel_games,
                game_count=game_count,  # Dec 30: Graduated thresholds
                harness_type=harness.value,  # Jan 9, 2026: Composite participant ID
                save_games_for_training=save_games,  # Jan 9, 2026: Save full training data
                recording_config=recording_config,  # Jan 13, 2026: Record gauntlet games
            )

        results = await asyncio.to_thread(run_gauntlet)

        # Use the GauntletResult's aggregated stats directly
        # The opponent_results dict has per-opponent stats, but we use totals from GauntletResult
        total_wins = results.total_wins
        total_losses = results.total_losses
        total_draws = results.total_draws
        total_games = results.total_games

        # Calculate win rate and estimate Elo
        if total_games == 0:
            raise ValueError("No games were played")

        win_rate = total_wins / total_games
        # Simple Elo estimation from win rate vs mixed baseline pool
        # Assumes baseline pool is approximately 1000 Elo
        baseline_elo = 1000
        if win_rate == 1.0:
            estimated_elo = baseline_elo + 400
        elif win_rate == 0.0:
            estimated_elo = baseline_elo - 400
        else:
            # Elo difference = 400 * log10(W/L)
            import math

            win_loss_ratio = win_rate / (1 - win_rate)
            elo_diff = 400 * math.log10(win_loss_ratio)
            estimated_elo = baseline_elo + elo_diff

        # Calculate confidence interval (simplified Wilson score)
        import math

        z = 1.96  # 95% confidence
        n = total_games
        p = win_rate
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
        ci_low = center - spread
        ci_high = center + spread

        # Convert win rate CI to Elo CI
        def win_rate_to_elo(wr: float) -> float:
            if wr >= 0.999:
                return baseline_elo + 400
            if wr <= 0.001:
                return baseline_elo - 400
            return baseline_elo + 400 * math.log10(wr / (1 - wr))

        elo_ci = (win_rate_to_elo(ci_low), win_rate_to_elo(ci_high))

        return EloRating(
            elo=estimated_elo,
            games_played=total_games,
            wins=total_wins,
            losses=total_losses,
            draws=total_draws,
            win_rate=win_rate,
            confidence_interval=elo_ci,
        )

    def _default_harness_config(self, harness: HarnessType) -> HarnessConfig:
        """Get default configuration for a harness type."""
        config = HarnessConfig(
            harness_type=harness,
            games_per_baseline=self.default_games_per_baseline,
            baselines=self.default_baselines.copy(),
        )

        # Set harness-specific defaults
        # Jan 2026: MCTS removed; use GUMBEL_MCTS or GPU_GUMBEL
        if harness in (HarnessType.GUMBEL_MCTS, HarnessType.GPU_GUMBEL):
            config.mcts_budget = 200
        elif harness in (HarnessType.MINIMAX, HarnessType.MAXN, HarnessType.BRS):
            config.search_depth = 4
        elif harness == HarnessType.POLICY_ONLY:
            config.temperature = 0.5

        return config

    async def evaluate_all_harnesses(
        self,
        model_path: str | Path,
        board_type: str,
        num_players: int,
        games_per_harness: int = 50,
        save_games: bool = True,
        register_with_elo: bool = True,
    ) -> dict[str, EloRating]:
        """Evaluate model under ALL compatible harnesses and record to EloService.

        This is a convenience method that evaluates a model under every compatible
        harness for the given model type and player count, records results to the
        EloService with composite participant IDs, and returns the ratings.

        Args:
            model_path: Path to model checkpoint
            board_type: Board configuration (e.g., "hex8", "square8")
            num_players: 2, 3, or 4
            games_per_harness: Games to play per harness (default 50)
            save_games: Whether to save games for training (default True)
            register_with_elo: Whether to register results with EloService (default True)

        Returns:
            Dict mapping harness_name -> EloRating

        Example:
            gauntlet = MultiHarnessGauntlet()
            ratings = await gauntlet.evaluate_all_harnesses(
                model_path="models/canonical_hex8_4p.pth",
                board_type="hex8",
                num_players=4,
                games_per_harness=50,
                save_games=True,
            )
            for harness, rating in ratings.items():
                print(f"  {harness}: {rating.display}")
        """
        # Create custom config with games_per_harness
        gauntlet = MultiHarnessGauntlet(
            default_games_per_baseline=games_per_harness // 2,  # Split across 2 baselines
            default_baselines=self.default_baselines,
            parallel_evaluations=self.parallel_evaluations,
        )

        # Run full evaluation
        result = await gauntlet.evaluate_model(
            model_path=model_path,
            board_type=board_type,
            num_players=num_players,
            save_games=save_games,
        )

        # Register with EloService if requested
        if register_with_elo:
            try:
                register_multi_harness_results(result)
            except (ImportError, RuntimeError, ValueError) as e:
                logger.warning(f"Failed to register with EloService: {e}")

        # Convert to dict[str, EloRating]
        ratings: dict[str, EloRating] = {}
        for harness, rating in result.harness_results.items():
            harness_name = harness.value if hasattr(harness, "value") else str(harness)
            ratings[harness_name] = rating

        return ratings


# ============================================
# Convenience Functions
# ============================================


async def evaluate_model_all_harnesses(
    model_path: str | Path,
    board_type: str = "square8",
    num_players: int = 2,
    model_type: str | None = None,
    games_per_baseline: int = 30,
) -> MultiHarnessResult:
    """Convenience function to evaluate a model under all compatible harnesses.

    Args:
        model_path: Path to model checkpoint
        board_type: Board type string
        num_players: Number of players
        model_type: Model type (auto-detected if None)
        games_per_baseline: Games per baseline opponent

    Returns:
        MultiHarnessResult with ratings for each harness
    """
    gauntlet = MultiHarnessGauntlet(default_games_per_baseline=games_per_baseline)
    return await gauntlet.evaluate_model(
        model_path=model_path,
        model_type=model_type,
        board_type=board_type,
        num_players=num_players,
    )


def get_harness_matrix() -> dict[str, list[str]]:
    """Get the harness compatibility matrix as plain strings.

    Returns:
        Dictionary mapping model types to compatible harness types
    """
    return {
        mt.value: [h.value for h in harnesses] for mt, harnesses in HARNESS_MATRIX.items()
    }


def register_multi_harness_results(
    result: MultiHarnessResult,
    elo_service: Any | None = None,
) -> dict[str, str]:
    """Register multi-harness evaluation results in the Elo system.

    December 30, 2025: Updated to use EloService.record_multi_harness_evaluation()
    for proper composite participant registration and Elo initialization.

    Args:
        result: MultiHarnessResult from evaluation
        elo_service: Optional EloService instance (lazy loaded if None)

    Returns:
        Dictionary mapping harness names to participant IDs
    """
    try:
        from app.training.elo_service import get_elo_service

        if elo_service is None:
            elo_service = get_elo_service()

        if not result.model_path or not result.board_type:
            logger.warning("Cannot register results: missing model_path or board_type")
            return {}

        # Convert harness_results to the format expected by record_multi_harness_evaluation
        harness_results_dict: dict[str, dict[str, Any]] = {}
        for harness, rating in result.harness_results.items():
            harness_name = harness.value if hasattr(harness, "value") else str(harness)
            harness_results_dict[harness_name] = {
                "elo": getattr(rating, "elo", 1500.0),
                "games_played": getattr(rating, "games_played", 0),
                "wins": getattr(rating, "wins", 0),
                "losses": getattr(rating, "losses", 0),
                "draws": getattr(rating, "draws", 0),
                "win_rate": getattr(rating, "win_rate", 0.0),
            }

        # Use the new unified method
        participant_ids = elo_service.record_multi_harness_evaluation(
            model_path=result.model_path,
            board_type=result.board_type,
            num_players=result.num_players,
            harness_results=harness_results_dict,
        )

        logger.info(
            f"Registered {len(participant_ids)} harness ratings for "
            f"{Path(result.model_path).stem}"
        )
        return participant_ids

    except ImportError as e:
        logger.warning(f"Could not register results: {e}")
        return {}


# ============================================
# Phase 1 Harness Integration (Dec 2025)
# ============================================


def get_harness_compatibility_from_registry(
    model_path: str | Path,
) -> dict[str, list[str]]:
    """Get harness compatibility using Phase 1 harness registry.

    This bridges the multi-harness gauntlet with the unified harness
    abstraction layer from Phase 1.

    Args:
        model_path: Path to model checkpoint

    Returns:
        Dictionary with model type and compatible harnesses
    """
    try:
        from app.ai.harness import (
            HarnessType as HT,
            ModelType as MT,
            get_compatible_harnesses,
        )

        path_str = str(model_path).lower()

        # Detect model type
        if "nnue" in path_str:
            model_type = MT.NNUE
        elif path_str.endswith(".pth") or path_str.endswith(".pt"):
            model_type = MT.NEURAL_NET
        else:
            model_type = MT.HEURISTIC

        compatible = get_compatible_harnesses(model_type)

        return {
            "model_type": model_type.value,
            "compatible_harnesses": [h.value for h in compatible],
        }
    except ImportError as e:
        logger.debug(f"Harness registry not available: {e}")
        return {"model_type": "unknown", "compatible_harnesses": []}


def create_harness_for_evaluation(
    harness_type: str,
    model_path: str | Path,
    board_type: str,
    num_players: int,
) -> Any:
    """Create a harness instance using Phase 1 factory.

    This uses the unified harness creation from the Phase 1 harness
    abstraction layer, providing consistent configuration.

    Args:
        harness_type: Harness type string (e.g., "gumbel_mcts")
        model_path: Path to model checkpoint
        board_type: Board type string
        num_players: Number of players

    Returns:
        AIHarness instance ready for evaluation
    """
    try:
        from app.ai.harness import HarnessType as HT, create_harness
        from app.models import BoardType as BT

        harness = create_harness(
            harness_type=HT(harness_type),
            model_path=model_path,
            board_type=BT(board_type),
            num_players=num_players,
        )
        return harness
    except ImportError as e:
        raise ImportError(f"Harness registry not available: {e}") from e


# Backwards-compatible aliases for existing code
run_multi_harness_evaluation = evaluate_model_all_harnesses


# ============================================
# Harness Compatibility Helpers (Dec 2025)
# Consolidated to use harness_registry as single source of truth
# ============================================

# Import canonical compatibility dict from harness_registry
from app.ai.harness.harness_registry import get_harness_compatibility_dict

# Backward compatibility: derive HARNESS_COMPATIBILITY from canonical source
HARNESS_COMPATIBILITY: dict[str, dict[str, Any]] = get_harness_compatibility_dict()


def get_compatible_harnesses_for_model(model_type: str) -> list[str]:
    """Get list of harnesses compatible with a model type.

    Args:
        model_type: Model type string ("nn" or "nnue")

    Returns:
        List of compatible harness type names
    """
    key = "nn" if model_type == "nn" else "nnue"
    compat_dict = get_harness_compatibility_dict()
    return [name for name, compat in compat_dict.items() if compat.get(key, False)]


def is_harness_compatible(harness_type: str, model_type: str) -> bool:
    """Check if a harness is compatible with a model type.

    Args:
        harness_type: Harness type string (e.g., "gumbel_mcts")
        model_type: Model type string ("nn" or "nnue")

    Returns:
        True if compatible, False otherwise
    """
    if not harness_type:
        return False
    compat_dict = get_harness_compatibility_dict()
    compat = compat_dict.get(harness_type)
    if compat is None:
        return False
    key = "nn" if model_type == "nn" else "nnue"
    return compat.get(key, False)


# ============================================
# NNUE-Specific Evaluation Profile (Dec 2025)
# Phase 4: NNUE Integration
# ============================================


@dataclass
class NNUEEvaluationProfile:
    """NNUE-specific evaluation profile for gauntlet testing.

    Configures how NNUE models should be evaluated under different
    harnesses based on player count:
    - 2-player: Minimax (alpha-beta with NNUE evaluation)
    - 3-4 player: MaxN or BRS with NNUE evaluation

    Attributes:
        board_type: Board type string (e.g., "hex8", "square8")
        num_players: Number of players (2, 3, or 4)
        minimax_depth: Search depth for minimax (2-player)
        maxn_depth: Search depth for MaxN (3-4 player)
        brs_depth: Search depth for BRS (3-4 player)
        games_per_baseline: Games to play against each baseline
        baselines: List of baseline opponent types
        include_brs: Whether to include BRS in 3-4 player evaluation
    """

    board_type: str = "square8"
    num_players: int = 2
    minimax_depth: int = 4
    maxn_depth: int = 3
    brs_depth: int = 3
    games_per_baseline: int = 30
    baselines: list[str] = field(default_factory=lambda: ["random", "heuristic"])
    include_brs: bool = True

    def get_harnesses(self) -> list[HarnessType]:
        """Get the harnesses to use for evaluation based on player count."""
        if self.num_players == 2:
            return [HarnessType.MINIMAX]
        else:
            harnesses = [HarnessType.MAXN]
            if self.include_brs:
                harnesses.append(HarnessType.BRS)
            return harnesses

    def get_harness_config(self, harness: HarnessType) -> HarnessConfig:
        """Get configuration for a specific harness."""
        if harness == HarnessType.MINIMAX:
            return HarnessConfig(
                harness_type=harness,
                games_per_baseline=self.games_per_baseline,
                baselines=self.baselines,
                search_depth=self.minimax_depth,
            )
        elif harness == HarnessType.MAXN:
            return HarnessConfig(
                harness_type=harness,
                games_per_baseline=self.games_per_baseline,
                baselines=self.baselines,
                search_depth=self.maxn_depth,
            )
        elif harness == HarnessType.BRS:
            return HarnessConfig(
                harness_type=harness,
                games_per_baseline=self.games_per_baseline,
                baselines=self.baselines,
                search_depth=self.brs_depth,
            )
        else:
            return HarnessConfig(
                harness_type=harness,
                games_per_baseline=self.games_per_baseline,
                baselines=self.baselines,
            )


async def evaluate_nnue_model(
    nnue_model_path: str | Path,
    board_type: str = "square8",
    num_players: int = 2,
    profile: NNUEEvaluationProfile | None = None,
) -> MultiHarnessResult:
    """Evaluate an NNUE model using appropriate harnesses.

    This is a convenience function that automatically selects the
    correct harnesses for NNUE evaluation based on player count:
    - 2-player: Minimax with alpha-beta pruning
    - 3-4 player: MaxN and optionally BRS

    Args:
        nnue_model_path: Path to NNUE model checkpoint
        board_type: Board type string (e.g., "hex8", "square8")
        num_players: Number of players (2, 3, or 4)
        profile: Optional evaluation profile (uses defaults if None)

    Returns:
        MultiHarnessResult with Elo ratings for each harness

    Example:
        # Evaluate 2-player NNUE under minimax
        result = await evaluate_nnue_model(
            "models/nnue/nnue_canonical_hex8_2p.pt",
            board_type="hex8",
            num_players=2,
        )
        print(f"Minimax Elo: {result.harness_results[HarnessType.MINIMAX].elo}")

        # Evaluate 4-player NNUE under MaxN and BRS
        result = await evaluate_nnue_model(
            "models/nnue/nnue_canonical_hex8_4p.pt",
            board_type="hex8",
            num_players=4,
        )
    """
    # Use default profile if none provided
    if profile is None:
        profile = NNUEEvaluationProfile(
            board_type=board_type,
            num_players=num_players,
        )

    # Get harnesses and configs
    harnesses = profile.get_harnesses()
    harness_configs = {h: profile.get_harness_config(h) for h in harnesses}

    # Determine model type based on player count
    model_type = ModelType.NNUE if num_players == 2 else ModelType.NNUE

    # Run evaluation
    gauntlet = MultiHarnessGauntlet(
        default_games_per_baseline=profile.games_per_baseline,
        default_baselines=profile.baselines,
    )

    result = await gauntlet.evaluate_model(
        model_path=nnue_model_path,
        model_type=model_type,
        board_type=board_type,
        num_players=num_players,
        harnesses=harnesses,
        harness_configs=harness_configs,
    )

    logger.info(
        f"NNUE evaluation complete: {Path(nnue_model_path).name} "
        f"({num_players}p, {board_type}) - Best: {result.best_harness} @ {result.best_elo:.0f} Elo"
    )

    return result


def get_nnue_evaluation_profile(
    board_type: str,
    num_players: int,
    *,
    fast: bool = False,
) -> NNUEEvaluationProfile:
    """Get a pre-configured NNUE evaluation profile.

    Args:
        board_type: Board type string
        num_players: Number of players
        fast: If True, use faster settings (fewer games, shallower search)

    Returns:
        Configured NNUEEvaluationProfile
    """
    if fast:
        return NNUEEvaluationProfile(
            board_type=board_type,
            num_players=num_players,
            minimax_depth=2,
            maxn_depth=2,
            brs_depth=2,
            games_per_baseline=10,
            baselines=["random", "heuristic"],
        )

    # Standard profile
    return NNUEEvaluationProfile(
        board_type=board_type,
        num_players=num_players,
        minimax_depth=4,
        maxn_depth=3,
        brs_depth=3,
        games_per_baseline=30,
        baselines=["random", "heuristic"],
    )
