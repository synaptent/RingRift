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

if TYPE_CHECKING:
    from app.models import BoardType


class ModelType(str, Enum):
    """Model architecture types."""

    NN = "nn"  # Full neural network (v2-v6)
    NNUE = "nnue"  # NNUE (scalar output, 2-player)
    NNUE_MP = "nnue_mp"  # Multi-player NNUE


class HarnessType(str, Enum):
    """AI algorithm harness types."""

    POLICY_ONLY = "policy_only"
    MCTS = "mcts"
    GUMBEL_MCTS = "gumbel_mcts"
    DESCENT = "descent"
    MINIMAX = "minimax"
    MAXN = "maxn"
    BRS = "brs"


# Compatibility matrix: which harnesses work with which model types
HARNESS_MATRIX: dict[ModelType, list[HarnessType]] = {
    ModelType.NN: [
        HarnessType.POLICY_ONLY,
        HarnessType.MCTS,
        HarnessType.GUMBEL_MCTS,
        HarnessType.DESCENT,
    ],
    ModelType.NNUE: [
        HarnessType.MINIMAX,
    ],
    ModelType.NNUE_MP: [
        HarnessType.MAXN,
        HarnessType.BRS,
        HarnessType.MINIMAX,  # Can use in 2p paranoid mode
    ],
}

# Player count restrictions for harnesses
HARNESS_PLAYER_RESTRICTIONS: dict[HarnessType, tuple[int, int]] = {
    HarnessType.POLICY_ONLY: (2, 4),
    HarnessType.MCTS: (2, 4),
    HarnessType.GUMBEL_MCTS: (2, 4),
    HarnessType.DESCENT: (2, 4),
    HarnessType.MINIMAX: (2, 2),  # Only 2-player
    HarnessType.MAXN: (3, 4),  # 3-4 player only
    HarnessType.BRS: (3, 4),  # 3-4 player only
}


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

    model_path: str
    model_type: ModelType
    board_type: str
    num_players: int
    harness_results: dict[HarnessType, EloRating]
    best_harness: HarnessType | None
    best_elo: float | None
    total_games: int
    evaluation_time_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_path": self.model_path,
            "model_type": self.model_type.value,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "harness_results": {
                h.value: {
                    "elo": r.elo,
                    "games_played": r.games_played,
                    "wins": r.wins,
                    "losses": r.losses,
                    "draws": r.draws,
                    "win_rate": r.win_rate,
                }
                for h, r in self.harness_results.items()
            },
            "best_harness": self.best_harness.value if self.best_harness else None,
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
        for harness in HARNESS_MATRIX.get(model_type, []):
            min_players, max_players = HARNESS_PLAYER_RESTRICTIONS.get(harness, (2, 4))
            if min_players <= num_players <= max_players:
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
            return ModelType.NNUE_MP
        if "nnue" in name:
            return ModelType.NNUE

        # Check checkpoint metadata
        try:
            from app.utils.torch_utils import safe_load_checkpoint

            checkpoint = safe_load_checkpoint(str(path), map_location="cpu")
            if isinstance(checkpoint, dict):
                if checkpoint.get("model_type") == "nnue_mp":
                    return ModelType.NNUE_MP
                if checkpoint.get("model_type") == "nnue":
                    return ModelType.NNUE
                if "num_players" in checkpoint and checkpoint["num_players"] >= 3:
                    # Multi-player NNUE if num_players stored
                    if "nnue" in name:
                        return ModelType.NNUE_MP
        except (FileNotFoundError, ValueError, RuntimeError):
            pass

        # Default to full NN
        return ModelType.NN

    async def evaluate_model(
        self,
        model_path: str | Path,
        model_type: ModelType | str | None = None,
        board_type: str = "square8",
        num_players: int = 2,
        harnesses: list[HarnessType | str] | None = None,
        harness_configs: dict[HarnessType, HarnessConfig] | None = None,
    ) -> MultiHarnessResult:
        """Evaluate a model under all compatible harnesses.

        Args:
            model_path: Path to model checkpoint
            model_type: Model type (auto-detected if None)
            board_type: Board type string (e.g., "hex8", "square8")
            num_players: Number of players
            harnesses: Specific harnesses to evaluate (None = all compatible)
            harness_configs: Override configurations per harness

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
        harness_results: dict[HarnessType, EloRating] = {}
        total_games = 0

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
    ) -> EloRating:
        """Evaluate model under a single harness.

        Args:
            model_path: Path to model checkpoint
            model_type: Model architecture type
            board_type: Board type string
            num_players: Number of players
            harness: Harness to evaluate under
            config: Harness configuration

        Returns:
            EloRating for this harness
        """
        from app.training.game_gauntlet import (
            BaselineOpponent,
            GauntletGameResult,
            run_baseline_gauntlet,
        )
        from app.models import AIType, BoardType as BT

        # Convert board type string to enum
        board_type_enum = BT(board_type)

        # Map harness to AIType for gauntlet
        harness_to_ai_type = {
            HarnessType.POLICY_ONLY: AIType.POLICY_ONLY,
            HarnessType.MCTS: AIType.MCTS,
            HarnessType.GUMBEL_MCTS: AIType.GUMBEL_MCTS,
            HarnessType.DESCENT: AIType.NN_DESCENT,
            HarnessType.MINIMAX: AIType.NNUE_MINIMAX,
            HarnessType.MAXN: AIType.MAXN,
            HarnessType.BRS: AIType.BRS,
        }

        ai_type = harness_to_ai_type.get(harness)
        if ai_type is None:
            raise ValueError(f"Unknown harness type: {harness}")

        # Map baseline strings to enum
        baseline_map = {
            "random": BaselineOpponent.RANDOM,
            "heuristic": BaselineOpponent.HEURISTIC,
            "weak_heuristic": BaselineOpponent.WEAK_HEURISTIC,
        }
        baselines = [baseline_map[b] for b in config.baselines if b in baseline_map]

        # Run gauntlet with this harness (in thread pool to not block)
        def run_gauntlet() -> dict[str, GauntletGameResult]:
            return run_baseline_gauntlet(
                model_path=str(model_path),
                board_type=board_type_enum,
                num_players=num_players,
                opponents=baselines,
                games_per_opponent=config.games_per_baseline,
                ai_type=ai_type,
                parallel_games=config.parallel_games,
                timeout=config.timeout_seconds,
            )

        results = await asyncio.to_thread(run_gauntlet)

        # Aggregate results across baselines
        total_wins = 0
        total_losses = 0
        total_draws = 0
        total_games = 0

        for baseline_result in results.values():
            total_wins += baseline_result.wins
            total_losses += baseline_result.losses
            total_draws += baseline_result.draws
            total_games += baseline_result.games_played

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
        if harness in (HarnessType.MCTS, HarnessType.GUMBEL_MCTS):
            config.mcts_budget = 200
        elif harness in (HarnessType.MINIMAX, HarnessType.MAXN, HarnessType.BRS):
            config.search_depth = 4
        elif harness == HarnessType.POLICY_ONLY:
            config.temperature = 0.5

        return config


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

    Args:
        result: MultiHarnessResult from evaluation
        elo_service: Optional EloService instance (lazy loaded if None)

    Returns:
        Dictionary mapping harness names to participant IDs
    """
    try:
        from app.training.composite_participant import make_composite_participant_id
        from app.training.elo_service import EloService

        if elo_service is None:
            elo_service = EloService.get_instance()

        participant_ids = {}
        model_name = Path(result.model_path).stem

        for harness, rating in result.harness_results.items():
            # Create composite participant ID
            participant_id = make_composite_participant_id(
                nn_id=model_name,
                ai_type=harness.value,
                config={"players": result.num_players},
            )

            # Register or update rating
            elo_service.register_participant(
                participant_id=participant_id,
                initial_elo=rating.elo,
                games_played=rating.games_played,
            )

            participant_ids[harness.value] = participant_id

        logger.info(
            f"Registered {len(participant_ids)} harness ratings for {model_name}"
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
