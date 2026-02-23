"""Tournament Scheduling Daemon - Automatic tournament execution on model events.

This daemon automatically schedules and runs tournaments when:
1. A new model is trained (TRAINING_COMPLETED event)
2. A model is promoted (MODEL_PROMOTED event)
3. Periodic ladder tournaments (configurable interval)

Features:
- Subscribes to training/promotion events
- Auto-schedules evaluation tournaments using RoundRobinScheduler
- Integrates with EloService for rating updates
- Supports gauntlet-style evaluation against baselines
- Emits EVALUATION_COMPLETED events for downstream processing

Usage:
    from app.coordination.tournament_daemon import (
        TournamentDaemon,
        TournamentDaemonConfig,
        get_tournament_daemon,
    )

    # Get singleton daemon
    daemon = get_tournament_daemon()

    # Start daemon (subscribes to events)
    await daemon.start()

    # Manually trigger evaluation
    await daemon.evaluate_model("path/to/model.pth", "hex8", 2)

    # Stop daemon
    await daemon.stop()
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# December 2025: Use consolidated daemon stats base class
from app.coordination.daemon_stats import EvaluationDaemonStats
from app.coordination.event_utils import make_config_key, parse_config_key
from app.coordination.event_emission_helpers import safe_emit_event
from app.training.composite_participant import extract_harness_type
from app.coordination.handler_base import HandlerBase
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
from app.ai.harness.base_harness import HarnessType, ModelType
from app.ai.harness.harness_registry import (
    HARNESS_PLAYER_RESTRICTIONS,
    is_harness_valid_for_player_count,
    get_harnesses_for_model_and_players,
)

# December 30, 2025: Game count for graduated thresholds
from app.utils.game_discovery import get_game_counts_summary

__all__ = [
    "TournamentDaemon",
    "TournamentDaemonConfig",
    "TournamentStats",
    "OpponentSpec",
    "OpponentDiversityMatrix",
    "GenerationTournamentScheduler",
    "TournamentTypeSelector",
    "get_tournament_daemon",
    "reset_tournament_daemon",
]


@dataclass
class TournamentDaemonConfig:
    """Configuration for the tournament daemon."""
    # Event subscriptions
    trigger_on_training_completed: bool = True
    trigger_on_model_promoted: bool = True  # Triggers cross-NN tournament after promotion (Jan 5, 2026)

    # Periodic tournaments
    enable_periodic_ladder: bool = True
    ladder_interval_seconds: float = 3600.0  # 1 hour

    # Evaluation settings
    # Feb 22, 2026: Raised from 20→50. With 20 games, 95% CI is ±22% win rate,
    # making promotion decisions statistically indistinguishable from random.
    games_per_evaluation: int = 50
    games_per_baseline: int = 20
    # Extended baselines for diverse Elo population (Dec 2025)
    # Includes random, heuristic variants, MCTS, and NNUE at different strengths
    baselines: list[str] = field(default_factory=lambda: [
        "random",           # ~400 Elo - baseline anchor
        "heuristic",        # ~1200 Elo - standard gate
        "heuristic_strong", # ~1400 Elo - difficulty 8
        "mcts_light",       # ~1500 Elo - 32 simulations
        "mcts_medium",      # ~1700 Elo - 128 simulations
    ])

    # Dec 31, 2025: NNUE baselines for architecture diversity
    # Added separately since they require NNUE model for the config
    nnue_baselines_2p: list[str] = field(default_factory=lambda: [
        "nnue_minimax_d4",  # ~1600 Elo - NNUE + minimax depth 4 (2-player only)
    ])
    nnue_baselines_mp: list[str] = field(default_factory=lambda: [
        "nnue_brs_d3",      # ~1550 Elo - NNUE + BRS depth 3 (3-4 player)
        "nnue_maxn_d3",     # ~1650 Elo - NNUE + MaxN depth 3 (3-4 player)
    ])
    enable_nnue_baselines: bool = True  # Test against NNUE baselines when available

    # Game recording for training data (Dec 2025)
    # When enabled, tournament games are saved to canonical databases for training
    enable_game_recording: bool = True
    recording_db_prefix: str = "tournament"
    recording_db_dir: str = "data/games"

    # Calibration tournaments - validate Elo ladder (Dec 2025)
    enable_calibration_tournaments: bool = True
    calibration_interval_seconds: float = 3600.0 * 24  # Daily
    calibration_games: int = 10  # Games per calibration matchup

    # Cross-NN version tournaments - compare model versions (Dec 2025)
    # Jan 4, 2026: Enabled - UnifiedNeuralNetFactory implemented in app/ai/neural_net.py
    enable_cross_nn_tournaments: bool = True
    cross_nn_interval_seconds: float = 3600.0 * 4  # Every 4 hours
    cross_nn_games_per_pairing: int = 20

    # Multi-harness evaluation - evaluate model under all compatible harnesses (Dec 31, 2025)
    # Tests model with MINIMAX, MAXN, BRS, GUMBEL_MCTS, etc. to find best (model, harness) combo
    enable_multi_harness_evaluation: bool = True
    multi_harness_games_per_baseline: int = 10
    # Only run multi-harness after model passes basic RANDOM/HEURISTIC gate
    multi_harness_min_baseline_winrate: float = 0.60  # 60% vs heuristic required

    # Jan 2026: Periodic multi-harness tournaments for regular diverse evaluation
    # Runs multi-harness evaluation on top models at regular intervals
    enable_periodic_multi_harness: bool = True
    periodic_multi_harness_interval_seconds: float = 3600.0 * 2  # Every 2 hours
    # Priority configs to evaluate first (most important for training)
    # Feb 2026: Include ALL 12 configs to prevent evaluation starvation
    multi_harness_priority_configs: list[str] = field(default_factory=lambda: [
        "hex8_2p", "hex8_3p", "hex8_4p",
        "square8_2p", "square8_3p", "square8_4p",
        "square19_2p", "square19_3p", "square19_4p",
        "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
    ])

    # Jan 2026: Stale evaluation detection - re-evaluate models with old ratings
    enable_stale_evaluation_detection: bool = True
    stale_evaluation_threshold_days: float = 1.0  # Re-evaluate if rating >1 day old (was 7)
    stale_evaluation_check_interval_seconds: float = 3600.0  # Check hourly

    # Jan 2026: Cross-config tournaments - compare models across board types
    # Validates transfer learning effectiveness and curriculum progression
    enable_cross_config_tournaments: bool = True
    cross_config_interval_seconds: float = 3600.0 * 6  # Every 6 hours
    cross_config_games_per_matchup: int = 10
    # Board family groups for cross-config comparison
    # Feb 2026: Include all 4 board families
    cross_config_families: list[list[str]] = field(default_factory=lambda: [
        ["hex8_2p", "hex8_3p", "hex8_4p"],  # Hex family
        ["square8_2p", "square8_3p", "square8_4p"],  # Square8 family
        ["square19_2p", "square19_3p", "square19_4p"],  # Square19 family
        ["hexagonal_2p", "hexagonal_3p", "hexagonal_4p"],  # Hexagonal family
    ])

    # Jan 2026: Top-N round-robin tournaments - all top models play each other
    # This generates high-quality training data from diverse model matchups
    enable_topn_roundrobin: bool = True
    topn_roundrobin_interval_seconds: float = 3600.0 * 4  # Every 4 hours
    topn_roundrobin_n: int = 10  # Top 10 models per config
    topn_roundrobin_games_per_matchup: int = 10  # 10 games per pairing
    topn_roundrobin_min_elo_games: int = 5  # Minimum games to be considered "rated"
    # Configs to run round-robin on (empty = all available configs)
    # Feb 2026: Include all 12 configs to prevent evaluation starvation
    topn_roundrobin_configs: list[str] = field(default_factory=lambda: [
        "hex8_2p", "hex8_3p", "hex8_4p",
        "square8_2p", "square8_3p", "square8_4p",
        "square19_2p", "square19_3p", "square19_4p",
        "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
    ])

    # Concurrency
    max_concurrent_games: int = 4

    # Timeouts
    game_timeout_seconds: float = 300.0  # 5 minutes per game
    evaluation_timeout_seconds: float = 1800.0  # 30 minutes per evaluation


@dataclass
class TournamentStats(EvaluationDaemonStats):
    """Statistics about tournament daemon activity.

    December 2025: Now extends EvaluationDaemonStats for consistent tracking.
    Inherits: evaluations_completed, evaluations_failed, games_played,
              models_evaluated, promotions_triggered, is_healthy(), etc.
    """

    # Tournament-specific fields
    event_triggers: int = 0

    # Backward compatibility aliases
    @property
    def tournaments_completed(self) -> int:
        """Alias for evaluations_completed (backward compatibility)."""
        return self.evaluations_completed

    @property
    def last_tournament_time(self) -> float:
        """Alias for last_evaluation_time (backward compatibility)."""
        return self.last_evaluation_time

    @property
    def errors(self) -> list[str]:
        """Return last error as list for backward compatibility."""
        if self.last_error:
            return [self.last_error]
        return []

    def record_tournament_success(self, games: int = 0) -> None:
        """Record a successful tournament."""
        self.record_evaluation_success(duration=0.0, games=games)

    def record_tournament_failure(self, error: str) -> None:
        """Record a failed tournament."""
        self.record_evaluation_failure(error)

    def record_event_trigger(self) -> None:
        """Record an event trigger."""
        self.event_triggers += 1
        self.evaluations_triggered += 1


# =============================================================================
# Opponent Diversity Matrix (January 2026)
# =============================================================================

@dataclass
class OpponentSpec:
    """Specification for a tournament opponent.

    Defines a specific opponent configuration including the model type,
    harness type (search algorithm), and difficulty level. Used to create
    diverse opponent pools for comprehensive model evaluation.

    Attributes:
        model_type: Type of model (NEURAL_NET, NNUE, HEURISTIC).
        harness_type: Search algorithm to use (GUMBEL_MCTS, MINIMAX, etc.).
        difficulty: Difficulty level 0.0-1.0 (maps to simulations/depth).
        model_id: Optional model identifier for NN/NNUE models.
        name_override: Optional custom name for this opponent.
    """

    model_type: ModelType
    harness_type: HarnessType
    difficulty: float = 1.0
    model_id: str | None = None
    name_override: str | None = None

    @property
    def name(self) -> str:
        """Generate a descriptive name for this opponent."""
        if self.name_override:
            return self.name_override
        difficulty_str = f"d{self.difficulty:.1f}" if self.difficulty != 1.0 else ""
        return f"{self.model_type.value}_{self.harness_type.value}{difficulty_str}"

    def is_valid_for_players(self, num_players: int) -> bool:
        """Check if this opponent is valid for the given player count."""
        return is_harness_valid_for_player_count(self.harness_type, num_players)


class OpponentDiversityMatrix:
    """Generates diverse opponent combinations for comprehensive evaluation.

    Creates opponent lists that cover:
    - All model types (NN, NNUE, heuristic)
    - All compatible harness types for each model type
    - Multiple difficulty levels for Elo calibration
    - Player-count restrictions (MINIMAX 2p only, MAXN/BRS 3-4p only)

    Usage:
        matrix = OpponentDiversityMatrix()
        opponents = matrix.get_opponents_for_config("hex8", 2, "standard")
        for opp in opponents:
            result = run_game(model, opp)
    """

    # Diversity levels determine how many opponents to generate
    DIVERSITY_LEVELS = {
        "minimal": 5,       # Quick sanity check
        "standard": 12,     # Balanced coverage
        "comprehensive": 25,  # Full matrix
    }

    # Difficulty levels for Elo calibration (maps to simulations/depth)
    DIFFICULTY_LEVELS = [0.25, 0.5, 0.75, 1.0]

    def __init__(self):
        """Initialize the diversity matrix."""
        self._logger = logging.getLogger(f"{__name__}.OpponentDiversityMatrix")

    def get_opponents_for_config(
        self,
        board_type: str,
        num_players: int,
        diversity_level: str = "standard",
        canonical_model_path: str | None = None,
    ) -> list[OpponentSpec]:
        """Generate diverse opponent list for a board configuration.

        Args:
            board_type: Board type (hex8, square8, etc.).
            num_players: Number of players (2, 3, or 4).
            diversity_level: One of "minimal", "standard", "comprehensive".
            canonical_model_path: Path to canonical NN model for this config.

        Returns:
            List of OpponentSpec objects for tournament evaluation.
        """
        max_opponents = self.DIVERSITY_LEVELS.get(diversity_level, 12)
        opponents: list[OpponentSpec] = []

        # 1. Baseline anchors (always included)
        opponents.extend(self._get_baseline_opponents())

        # 2. NN opponents (if model available)
        if canonical_model_path:
            opponents.extend(
                self._get_nn_opponents(num_players, canonical_model_path)
            )

        # 3. NNUE opponents (player-count aware)
        opponents.extend(self._get_nnue_opponents(num_players))

        # 4. Heuristic opponents at various difficulties
        opponents.extend(self._get_heuristic_opponents())

        # Filter by player count validity
        valid_opponents = [
            opp for opp in opponents
            if opp.is_valid_for_players(num_players)
        ]

        # Trim to diversity level limit
        if len(valid_opponents) > max_opponents:
            valid_opponents = valid_opponents[:max_opponents]

        self._logger.info(
            f"Generated {len(valid_opponents)} opponents for {board_type}_{num_players}p "
            f"(diversity={diversity_level})"
        )

        return valid_opponents

    def _get_baseline_opponents(self) -> list[OpponentSpec]:
        """Get baseline anchor opponents (RANDOM, HEURISTIC)."""
        return [
            OpponentSpec(
                model_type=ModelType.HEURISTIC,
                harness_type=HarnessType.RANDOM,
                difficulty=1.0,
                name_override="random_baseline",
            ),
            OpponentSpec(
                model_type=ModelType.HEURISTIC,
                harness_type=HarnessType.HEURISTIC,
                difficulty=0.5,
                name_override="heuristic_weak",
            ),
            OpponentSpec(
                model_type=ModelType.HEURISTIC,
                harness_type=HarnessType.HEURISTIC,
                difficulty=1.0,
                name_override="heuristic_strong",
            ),
        ]

    def _get_nn_opponents(
        self,
        num_players: int,
        model_path: str,
    ) -> list[OpponentSpec]:
        """Get neural network opponents with various harnesses."""
        opponents = []

        # NN with Gumbel MCTS at different simulation budgets
        for difficulty in [0.5, 1.0]:
            opponents.append(OpponentSpec(
                model_type=ModelType.NEURAL_NET,
                harness_type=HarnessType.GUMBEL_MCTS,
                difficulty=difficulty,
                model_id=model_path,
            ))

        # GPU Gumbel (high throughput variant)
        opponents.append(OpponentSpec(
            model_type=ModelType.NEURAL_NET,
            harness_type=HarnessType.GPU_GUMBEL,
            difficulty=1.0,
            model_id=model_path,
        ))

        # Policy-only (no search, pure network)
        opponents.append(OpponentSpec(
            model_type=ModelType.NEURAL_NET,
            harness_type=HarnessType.POLICY_ONLY,
            difficulty=1.0,
            model_id=model_path,
        ))

        # Descent (gradient-based move selection)
        opponents.append(OpponentSpec(
            model_type=ModelType.NEURAL_NET,
            harness_type=HarnessType.DESCENT,
            difficulty=1.0,
            model_id=model_path,
        ))

        return opponents

    def _get_nnue_opponents(self, num_players: int) -> list[OpponentSpec]:
        """Get NNUE opponents with all compatible harnesses.

        All three NNUE-compatible harnesses (MINIMAX, MAXN, BRS) work for
        2-4 players. They differ in search strategy:
        - MINIMAX (paranoid): Assumes opponents cooperate against us
        - MAXN: Each player maximizes their own score
        - BRS: Best-Reply Search - greedy multiplayer approximation
        """
        opponents = []
        nnue_harnesses = [HarnessType.MINIMAX, HarnessType.MAXN, HarnessType.BRS]

        for harness in nnue_harnesses:
            for difficulty in [0.5, 1.0]:
                harness_name = harness.value.lower()
                opponents.append(OpponentSpec(
                    model_type=ModelType.NNUE,
                    harness_type=harness,
                    difficulty=difficulty,
                    name_override=f"nnue_{harness_name}_d{difficulty:.1f}",
                ))

        return opponents

    def _get_heuristic_opponents(self) -> list[OpponentSpec]:
        """Get pure heuristic opponents at various difficulties."""
        opponents = []

        for difficulty in self.DIFFICULTY_LEVELS:
            opponents.append(OpponentSpec(
                model_type=ModelType.HEURISTIC,
                harness_type=HarnessType.HEURISTIC,
                difficulty=difficulty,
            ))

        return opponents

    def get_standard_matrix(self, num_players: int) -> list[OpponentSpec]:
        """Get the standard diversity matrix for any player count.

        This is the recommended opponent set for regular evaluation.

        Args:
            num_players: Number of players (2, 3, or 4).

        Returns:
            List of OpponentSpec for standard evaluation.
        """
        return self.get_opponents_for_config(
            board_type="generic",
            num_players=num_players,
            diversity_level="standard",
        )


# =============================================================================
# Generation Tournament Scheduler (January 2026)
# =============================================================================

class GenerationTournamentScheduler:
    """Schedules head-to-head tournaments between model generations.

    Finds parent-child generation pairs that haven't been compared
    and schedules tournaments to validate training improvement.

    Usage:
        from app.coordination.generation_tracker import get_generation_tracker
        tracker = get_generation_tracker()
        scheduler = GenerationTournamentScheduler(tracker)
        pairs = scheduler.get_untested_pairs()
        for parent, child in pairs:
            result = scheduler.run_generation_tournament(parent, child)
    """

    def __init__(self, generation_tracker: Any | None = None):
        """Initialize the scheduler.

        Args:
            generation_tracker: Optional GenerationTracker instance.
                If not provided, will be lazily loaded when needed.
        """
        self._tracker = generation_tracker
        self._logger = logging.getLogger(f"{__name__}.GenerationTournamentScheduler")

    @property
    def tracker(self) -> Any:
        """Lazy-load the generation tracker if not provided."""
        if self._tracker is None:
            from app.coordination.generation_tracker import get_generation_tracker
            self._tracker = get_generation_tracker()
        return self._tracker

    def get_untested_pairs(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> list[tuple[Any, Any]]:
        """Find parent-child pairs without tournament results.

        Args:
            board_type: Optional filter by board type.
            num_players: Optional filter by player count.

        Returns:
            List of (parent_gen, child_gen) tuples that need tournaments.
        """
        try:
            all_gens = self.tracker.get_all_generations(
                board_type=board_type,
                num_players=num_players,
            )
        except Exception as e:
            self._logger.error(f"Failed to get generations: {e}")
            return []

        # Build lookup by generation_id
        gen_by_id = {g.generation_id: g for g in all_gens}
        untested: list[tuple[Any, Any]] = []

        for gen in all_gens:
            if gen.parent_generation is None:
                continue

            parent = gen_by_id.get(gen.parent_generation)
            if parent is None:
                continue

            # Check if both model files exist
            parent_exists = parent.model_path and Path(parent.model_path).exists()
            child_exists = gen.model_path and Path(gen.model_path).exists()

            if not parent_exists or not child_exists:
                continue

            # Check if tournament already exists
            try:
                tournaments = self.tracker.get_tournaments_for_generation(
                    gen.generation_id
                )
                has_parent_match = any(
                    (t.gen_a == gen.generation_id and t.gen_b == parent.generation_id) or
                    (t.gen_b == gen.generation_id and t.gen_a == parent.generation_id)
                    for t in tournaments
                )
                if not has_parent_match:
                    untested.append((parent, gen))
            except Exception as e:
                self._logger.warning(
                    f"Error checking tournaments for gen {gen.generation_id}: {e}"
                )

        self._logger.info(f"Found {len(untested)} untested generation pairs")
        return untested

    def get_tournament_priority(
        self,
        parent: Any,
        child: Any,
    ) -> float:
        """Calculate priority score for a generation pair tournament.

        Higher priority for:
        - More recent generations (newer training)
        - Larger training sample counts
        - Configs with less tournament coverage

        Args:
            parent: Parent generation info.
            child: Child generation info.

        Returns:
            Priority score (higher = more important).
        """
        priority = 0.0

        # Recency bonus (newer = higher priority)
        priority += child.generation_id * 0.1

        # Training data bonus
        if child.training_samples:
            priority += min(child.training_samples / 100000, 1.0) * 0.3

        # Version bump bonus (v2 vs v1 is interesting)
        if child.version > parent.version:
            priority += 0.5

        return priority


# =============================================================================
# Tournament Type Selector (January 2026)
# =============================================================================

class TournamentTypeSelector:
    """Selects tournament type based on current evaluation needs.

    Uses weighted random selection with dynamic priority adjustments
    based on system state (untested generations, model coverage, etc.).

    Tournament types:
    - generation_head_to_head: Validate training improvement
    - multi_harness_eval: Test robustness across harnesses
    - baseline_gauntlet: Standard baseline evaluation
    - calibration: Elo gap validation
    - stale_reevaluation: Refresh old ratings

    Usage:
        selector = TournamentTypeSelector()
        context = {"untested_generation_pairs": 5}
        tournament_type = selector.select_type(context)
    """

    DEFAULT_WEIGHTS = {
        "generation_head_to_head": 0.30,  # Validate training progress
        "multi_harness_eval": 0.25,       # Harness robustness
        "baseline_gauntlet": 0.20,        # Standard evaluation
        "calibration": 0.15,              # Elo gap validation
        "stale_reevaluation": 0.10,       # Refresh old evaluations
    }

    def __init__(self, weights: dict[str, float] | None = None):
        """Initialize the selector.

        Args:
            weights: Optional custom weights. Uses DEFAULT_WEIGHTS if not provided.
        """
        self._weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._logger = logging.getLogger(f"{__name__}.TournamentTypeSelector")

    def select_type(self, context: dict[str, Any] | None = None) -> str:
        """Select a tournament type based on context.

        Args:
            context: Optional context dict with keys:
                - untested_generation_pairs: Count of pairs without tournaments
                - models_lacking_harness_coverage: Models needing multi-harness
                - stale_model_count: Models with old ratings
                - calibration_needed: Whether Elo calibration is needed

        Returns:
            Selected tournament type string.
        """
        import random

        weights = self._weights.copy()

        if context:
            # Boost generation tournaments when untested pairs exist
            if context.get("untested_generation_pairs", 0) > 0:
                weights["generation_head_to_head"] += 0.2

            # Boost multi-harness when models lack coverage
            if context.get("models_lacking_harness_coverage", 0) > 0:
                weights["multi_harness_eval"] += 0.15

            # Boost stale reevaluation when many old ratings
            if context.get("stale_model_count", 0) > 5:
                weights["stale_reevaluation"] += 0.15

            # Boost calibration when explicitly needed
            if context.get("calibration_needed", False):
                weights["calibration"] += 0.20

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # Weighted random selection
        types = list(weights.keys())
        probs = [weights[t] for t in types]
        selected = random.choices(types, weights=probs, k=1)[0]

        self._logger.debug(
            f"Selected tournament type: {selected} "
            f"(weights: {weights}, context: {context})"
        )

        return selected

    def get_weights(self) -> dict[str, float]:
        """Get current weights (copy)."""
        return self._weights.copy()

    def set_weight(self, tournament_type: str, weight: float) -> None:
        """Set weight for a tournament type.

        Args:
            tournament_type: Tournament type name.
            weight: New weight (0.0-1.0).
        """
        if tournament_type in self._weights:
            self._weights[tournament_type] = max(0.0, min(1.0, weight))


class TournamentDaemon(HandlerBase):
    """Daemon that automatically schedules tournaments based on events.

    Subscribes to training/promotion events and triggers appropriate
    evaluation tournaments using the existing tournament infrastructure.

    January 2026: Migrated to HandlerBase for unified lifecycle and singleton.
    Uses _on_start() to spawn multiple background tasks with different intervals.
    """

    def __init__(self, config: TournamentDaemonConfig | None = None):
        """Initialize the tournament daemon.

        Args:
            config: Daemon configuration
        """
        self.config = config or TournamentDaemonConfig()
        self.node_id = socket.gethostname()

        # Initialize HandlerBase - use evaluation queue check interval as cycle
        super().__init__(
            name=f"tournament_daemon_{self.node_id}",
            cycle_interval=10.0,  # Check evaluation queue every 10s
        )

        # Background tasks for different intervals
        self._periodic_task: asyncio.Task | None = None
        self._calibration_task: asyncio.Task | None = None  # Dec 2025
        self._cross_nn_task: asyncio.Task | None = None  # Dec 2025
        self._multi_harness_task: asyncio.Task | None = None  # Jan 2026
        self._stale_eval_task: asyncio.Task | None = None  # Jan 2026
        self._cross_config_task: asyncio.Task | None = None  # Jan 2026
        self._topn_roundrobin_task: asyncio.Task | None = None  # Jan 2026
        self._tournament_stats = TournamentStats()
        self._evaluation_queue: asyncio.Queue = asyncio.Queue()

        # Jan 2026: Track last evaluation times per config for stale detection
        self._last_harness_eval_time: dict[str, float] = {}

        # Tracking attributes for health_check and get_status (Jan 2026)
        self._cycle_count: int = 0
        self._recent_errors: list[str] = []

    def _get_event_subscriptions(self) -> dict:
        """Get declarative event subscriptions (HandlerBase pattern).

        Returns:
            Dict mapping event names to handler methods
        """
        subscriptions = {}

        if self.config.trigger_on_training_completed:
            subscriptions["TRAINING_COMPLETED"] = self._on_training_completed

        if self.config.trigger_on_model_promoted:
            subscriptions["MODEL_PROMOTED"] = self._on_model_promoted

        return subscriptions

    async def _on_start(self) -> None:
        """Hook called after HandlerBase start (spawns background tasks)."""
        # Start periodic ladder tournaments
        if self.config.enable_periodic_ladder:
            self._periodic_task = self._safe_create_task(
                self._periodic_ladder_loop(),
                context="tournament_periodic_ladder"
            )

        # Start calibration tournaments (Dec 2025)
        if self.config.enable_calibration_tournaments:
            self._calibration_task = self._safe_create_task(
                self._calibration_loop(),
                context="tournament_calibration"
            )

        # Start cross-NN version tournaments (Dec 2025)
        if self.config.enable_cross_nn_tournaments:
            self._cross_nn_task = self._safe_create_task(
                self._cross_nn_loop(),
                context="tournament_cross_nn"
            )

        # Jan 2026: Start periodic multi-harness evaluation loop
        if self.config.enable_periodic_multi_harness:
            self._multi_harness_task = self._safe_create_task(
                self._periodic_multi_harness_loop(),
                context="tournament_periodic_multi_harness"
            )

        # Jan 2026: Start stale evaluation detection loop
        if self.config.enable_stale_evaluation_detection:
            self._stale_eval_task = self._safe_create_task(
                self._stale_evaluation_loop(),
                context="tournament_stale_eval"
            )

        # Jan 2026: Start cross-config tournament loop
        if self.config.enable_cross_config_tournaments:
            self._cross_config_task = self._safe_create_task(
                self._cross_config_loop(),
                context="tournament_cross_config"
            )

        # Jan 2026: Start top-N round-robin tournament loop
        if self.config.enable_topn_roundrobin:
            self._topn_roundrobin_task = self._safe_create_task(
                self._periodic_topn_roundrobin_loop(),
                context="tournament_topn_roundrobin"
            )

        logger.info(
            f"TournamentDaemon started (periodic_ladder={self.config.enable_periodic_ladder}, "
            f"calibration={self.config.enable_calibration_tournaments}, "
            f"cross_nn={self.config.enable_cross_nn_tournaments}, "
            f"multi_harness={self.config.enable_periodic_multi_harness}, "
            f"topn_roundrobin={self.config.enable_topn_roundrobin}, "
            f"stale_eval={self.config.enable_stale_evaluation_detection}, "
            f"cross_config={self.config.enable_cross_config_tournaments})"
        )

    async def _on_stop(self) -> None:
        """Hook called before HandlerBase stop (cancels background tasks)."""
        # Cancel periodic task
        if self._periodic_task:
            self._periodic_task.cancel()
            try:
                await asyncio.wait_for(self._periodic_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._periodic_task = None

        # Cancel calibration task (Dec 2025)
        if self._calibration_task:
            self._calibration_task.cancel()
            try:
                await asyncio.wait_for(self._calibration_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._calibration_task = None

        # Cancel cross-NN task (Dec 2025)
        if self._cross_nn_task:
            self._cross_nn_task.cancel()
            try:
                await asyncio.wait_for(self._cross_nn_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._cross_nn_task = None

        # Cancel multi-harness task (Jan 2026)
        if self._multi_harness_task:
            self._multi_harness_task.cancel()
            try:
                await asyncio.wait_for(self._multi_harness_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._multi_harness_task = None

        # Cancel stale evaluation task (Jan 2026)
        if self._stale_eval_task:
            self._stale_eval_task.cancel()
            try:
                await asyncio.wait_for(self._stale_eval_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._stale_eval_task = None

        # Cancel cross-config task (Jan 2026)
        if self._cross_config_task:
            self._cross_config_task.cancel()
            try:
                await asyncio.wait_for(self._cross_config_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._cross_config_task = None

        # Cancel top-N round-robin task (Jan 2026)
        if self._topn_roundrobin_task:
            self._topn_roundrobin_task.cancel()
            try:
                await asyncio.wait_for(self._topn_roundrobin_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._topn_roundrobin_task = None

        logger.info("TournamentDaemon stopped")

    async def _run_cycle(self) -> None:
        """Process evaluation queue (HandlerBase main work loop).

        Called every cycle_interval (10s) to check for pending evaluations.
        """
        try:
            # Non-blocking check for evaluation request
            try:
                request = self._evaluation_queue.get_nowait()
            except asyncio.QueueEmpty:
                return  # Nothing to process

            # Run evaluation
            await self._run_evaluation(
                model_path=request["model_path"],
                board_type=request["board_type"],
                num_players=request["num_players"],
                trigger=request.get("trigger", "unknown"),
            )
            self._record_success()

        except Exception as e:
            logger.error(f"Error processing evaluation: {e}")
            self._record_error(str(e))

    async def start(self) -> None:
        """Start the tournament daemon (delegates to HandlerBase)."""
        await super().start()

    async def stop(self) -> None:
        """Stop the tournament daemon (delegates to HandlerBase)."""
        await super().stop()

    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._running


    def _on_training_completed(self, event) -> None:
        """Handle TRAINING_COMPLETED event."""
        self._tournament_stats.event_triggers += 1

        payload = getattr(event, "payload", {}) or {}
        model_path = payload.get("model_path")
        config = payload.get("config", "")
        success = payload.get("success", True)

        if not success:
            logger.debug(f"Skipping evaluation for failed training: {config}")
            return

        if not model_path:
            logger.warning(f"TRAINING_COMPLETED event missing model_path: {config}")
            return

        # Parse config
        board_type, num_players = self._parse_config(config)
        if not board_type:
            logger.warning(f"Could not parse config: {config}")
            return

        logger.info(f"Training completed for {config}, queueing evaluation")

        # Queue evaluation
        self._evaluation_queue.put_nowait({
            "model_path": model_path,
            "board_type": board_type,
            "num_players": num_players,
            "trigger": "training_completed",
        })

    def _on_model_promoted(self, event) -> None:
        """Handle MODEL_PROMOTED event.

        Jan 5, 2026: Now triggers cross-NN tournament after promotion to compare
        newly promoted model against previous model versions.
        """
        self._tournament_stats.event_triggers += 1

        payload = getattr(event, "payload", {}) or {}
        model_id = payload.get("model_id")
        config_key = payload.get("config_key") or payload.get("config", "")
        model_path = payload.get("model_path", "")

        logger.info(f"Model promoted: {model_id} ({config_key})")

        # Check if cross-NN tournaments are enabled after promotion
        if not self.config.trigger_on_model_promoted:
            return

        if not self.config.enable_cross_nn_tournaments:
            logger.debug("Cross-NN tournaments disabled, skipping post-promotion tournament")
            return

        # Parse config to get board_type and num_players
        board_type, num_players = self._parse_config(config_key)
        if not board_type:
            logger.warning(f"Could not parse config for post-promotion tournament: {config_key}")
            return

        logger.info(
            f"Scheduling cross-NN tournament for {config_key} after promotion "
            f"(model: {model_path or model_id})"
        )

        # Schedule cross-NN tournament via fire-and-forget task
        # This runs the tournament comparing the newly promoted model against previous versions
        self._safe_create_task(
            self._run_cross_nn_tournament(),
            context=f"post_promotion_cross_nn_{config_key}",
        )

    def _parse_config(self, config: str) -> tuple[str | None, int | None]:
        """Parse board_type and num_players from config string.

        December 30, 2025: Migrated to use consolidated parse_config_key utility.
        """
        parsed = parse_config_key(config)
        if parsed:
            return parsed.board_type, parsed.num_players
        return None, None


    async def _periodic_ladder_loop(self) -> None:
        """Periodic ladder tournament loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.ladder_interval_seconds)

                if not self._running:
                    break

                logger.info("Running periodic ladder tournament")
                await self._run_ladder_tournament()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic ladder: {e}")
                self._tournament_stats.record_failure(str(e))

    async def _calibration_loop(self) -> None:
        """Periodic calibration tournament loop (Dec 2025).

        Runs calibration tournaments at configured interval to validate
        that Elo ladder gaps match expected win rates.
        """
        while self._running:
            try:
                await asyncio.sleep(self.config.calibration_interval_seconds)

                if not self._running:
                    break

                logger.info("Running scheduled calibration tournament")
                results = await self._run_calibration_tournament()

                if results.get("all_valid"):
                    logger.info("Calibration tournament: all matchups valid")
                else:
                    logger.warning(f"Calibration tournament: some matchups invalid")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in calibration loop: {e}")
                self._tournament_stats.record_failure(str(e))

    async def _cross_nn_loop(self) -> None:
        """Periodic cross-NN version tournament loop (Dec 2025).

        Runs tournaments between different NN versions (v2, v3, v4, etc.)
        to track model evolution and maintain accurate Elo ratings.
        """
        while self._running:
            try:
                await asyncio.sleep(self.config.cross_nn_interval_seconds)

                if not self._running:
                    break

                logger.info("Running scheduled cross-NN tournament")
                results = await self._run_cross_nn_tournament()

                if results.get("success"):
                    games_played = results.get("games_played", 0)
                    logger.info(f"Cross-NN tournament completed: {games_played} games")
                else:
                    logger.warning(f"Cross-NN tournament failed: {results.get('error')}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cross-NN loop: {e}")
                self._tournament_stats.record_failure(str(e))

    async def _run_evaluation(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        trigger: str = "manual",
    ) -> dict[str, Any]:
        """Run evaluation for a trained model.

        Args:
            model_path: Path to model checkpoint
            board_type: Board type
            num_players: Number of players
            trigger: What triggered this evaluation

        Returns:
            Evaluation results dict
        """
        self._tournament_stats.evaluations_triggered += 1
        start_time = time.time()

        logger.info(f"Starting evaluation: {model_path} ({board_type}_{num_players}p)")

        results = {
            "model_path": model_path,
            "board_type": board_type,
            "num_players": num_players,
            "trigger": trigger,
            "success": False,
            "win_rates": {},
            "elo": None,
            "games_played": 0,
            "duration_seconds": 0.0,
        }

        try:
            # Run gauntlet evaluation
            from app.training.game_gauntlet import BaselineOpponent, run_baseline_gauntlet

            # Create recording config if enabled (Dec 2025 - tournament games for training)
            recording_config = None
            if self.config.enable_game_recording:
                try:
                    from app.db.unified_recording import RecordingConfig, RecordSource
                    recording_config = RecordingConfig(
                        board_type=board_type,
                        num_players=num_players,
                        source=RecordSource.TOURNAMENT,
                        engine_mode="gauntlet",
                        db_prefix=self.config.recording_db_prefix,
                        db_dir=self.config.recording_db_dir,
                        store_history_entries=True,
                    )
                    logger.debug(f"Tournament recording enabled for {board_type}_{num_players}p")
                except ImportError:
                    logger.debug("Recording module not available, games will not be saved")

            # Dec 30, 2025: Get game count for graduated thresholds
            config_key = make_config_key(board_type, num_players)
            try:
                game_counts = get_game_counts_summary()
                game_count = game_counts.get(config_key, 0)
            except (OSError, RuntimeError):
                game_count = None  # Will use fallback thresholds

            gauntlet_results = await asyncio.wait_for(
                asyncio.to_thread(
                    run_baseline_gauntlet,
                    model_path=model_path,
                    board_type=board_type,
                    num_players=num_players,
                    games_per_opponent=self.config.games_per_baseline,
                    opponents=[BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC],
                    recording_config=recording_config,
                    game_count=game_count,  # Dec 30: Graduated thresholds
                ),
                timeout=self.config.evaluation_timeout_seconds,
            )

            results["success"] = True
            results["games_played"] = gauntlet_results.total_games
            results["win_rates"] = {
                opponent: stats.get("win_rate", 0.0)
                for opponent, stats in gauntlet_results.opponent_results.items()
            }

            # Update ELO
            if gauntlet_results.estimated_elo:
                results["elo"] = gauntlet_results.estimated_elo
                await self._update_elo(model_path, board_type, num_players, gauntlet_results)

            self._tournament_stats.games_played += results["games_played"]
            self._tournament_stats.last_evaluation_time = time.time()

            logger.info(
                f"Evaluation complete: {model_path} - "
                f"win_rates={results['win_rates']}, elo={results['elo']}"
            )

            # Dec 31, 2025: Multi-harness evaluation for harness diversity
            # Only run if model passed the baseline gate (60%+ vs heuristic)
            heuristic_winrate = results["win_rates"].get("heuristic", 0.0)
            if (
                self.config.enable_multi_harness_evaluation
                and heuristic_winrate >= self.config.multi_harness_min_baseline_winrate
            ):
                try:
                    multi_harness_results = await self._run_multi_harness_evaluation(
                        model_path, board_type, num_players
                    )
                    results["multi_harness"] = multi_harness_results
                    logger.info(
                        f"Multi-harness evaluation complete: best_harness={multi_harness_results.get('best_harness')}, "
                        f"best_elo={multi_harness_results.get('best_elo')}"
                    )
                except Exception as e:
                    logger.warning(f"Multi-harness evaluation failed (non-fatal): {e}")
                    results["multi_harness_error"] = str(e)
            elif self.config.enable_multi_harness_evaluation:
                logger.debug(
                    f"Skipping multi-harness: heuristic winrate {heuristic_winrate:.1%} < "
                    f"{self.config.multi_harness_min_baseline_winrate:.1%} threshold"
                )

            # Dec 31, 2025: NNUE baseline evaluation for architecture diversity
            # Tests model against NNUE-based baselines when NNUE model exists for config
            if self.config.enable_nnue_baselines and heuristic_winrate >= 0.50:
                try:
                    nnue_results = await self._run_nnue_baseline_evaluation(
                        model_path, board_type, num_players, recording_config
                    )
                    if nnue_results:
                        results["nnue_baselines"] = nnue_results
                        logger.info(
                            f"NNUE baseline evaluation complete: {nnue_results}"
                        )
                except Exception as e:
                    logger.warning(f"NNUE baseline evaluation failed (non-fatal): {e}")
                    results["nnue_baseline_error"] = str(e)

        except asyncio.TimeoutError:
            logger.error(f"Evaluation timeout: {model_path}")
            results["error"] = "timeout"
            self._tournament_stats.record_failure(f"Evaluation timeout: {model_path}")

        except ImportError as e:
            logger.warning(f"GameGauntlet not available: {e}")
            # Fall back to basic match execution
            results = await self._run_basic_evaluation(
                model_path, board_type, num_players
            )

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            results["error"] = str(e)
            self._tournament_stats.record_failure(str(e))

        results["duration_seconds"] = time.time() - start_time

        # Emit EVALUATION_COMPLETED event
        await self._emit_evaluation_completed(results)

        return results

    async def _run_basic_evaluation(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
    ) -> dict[str, Any]:
        """Run basic evaluation when GameGauntlet is not available.

        Uses the scheduler directly for match generation.
        """
        results = {
            "model_path": model_path,
            "board_type": board_type,
            "num_players": num_players,
            "success": False,
            "win_rates": {},
            "games_played": 0,
        }

        try:
            from app.models import BoardType
            from app.tournament.scheduler import RoundRobinScheduler

            # Create scheduler
            scheduler = RoundRobinScheduler(
                games_per_pairing=2,
                shuffle_order=True,
            )

            # Generate matches against baselines
            model_id = Path(model_path).stem
            agents = [model_id] + self.config.baselines

            board_type_enum = BoardType(board_type)
            matches = scheduler.generate_matches(
                agent_ids=agents,
                board_type=board_type_enum,
                num_players=num_players,
            )

            logger.info(f"Generated {len(matches)} matches for basic evaluation")

            # Note: Match execution would require game engine integration
            # For now, just report the scheduled matches
            results["scheduled_matches"] = len(matches)
            results["success"] = True

        except Exception as e:
            logger.error(f"Basic evaluation failed: {e}")
            results["error"] = str(e)

        return results

    async def _run_multi_harness_evaluation(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
    ) -> dict[str, Any]:
        """Run multi-harness evaluation to find best (model, harness) combo.

        December 31, 2025: Integrates MultiHarnessGauntlet for harness diversity.
        Tests model under MINIMAX, MAXN, BRS, GUMBEL_MCTS, etc.

        Args:
            model_path: Path to model checkpoint
            board_type: Board type
            num_players: Number of players

        Returns:
            Dict with per-harness Elo ratings and best harness
        """
        from app.training.multi_harness_gauntlet import MultiHarnessGauntlet

        gauntlet = MultiHarnessGauntlet(
            default_games_per_baseline=self.config.multi_harness_games_per_baseline,
            default_baselines=["random", "heuristic"],
        )

        # Run evaluation under all compatible harnesses
        result = await gauntlet.evaluate_model(
            model_path=model_path,
            board_type=board_type,
            num_players=num_players,
        )

        # Extract per-harness results for tracking
        harness_elos = {}
        for harness_type, elo_rating in result.harness_results.items():
            harness_name = harness_type.value if hasattr(harness_type, "value") else str(harness_type)
            harness_elos[harness_name] = {
                "elo": elo_rating.elo,
                "win_rate": elo_rating.win_rate,
                "games": elo_rating.games_played,
            }

            # Emit per-harness EVALUATION_COMPLETED event for Elo tracking
            await self._emit_harness_evaluation_completed(
                model_path=model_path,
                board_type=board_type,
                num_players=num_players,
                harness_type=harness_name,
                elo=elo_rating.elo,
                win_rate=elo_rating.win_rate,
            )

        return {
            "best_harness": result.best_harness.value if result.best_harness else None,
            "best_elo": result.best_elo,
            "total_games": result.total_games,
            "evaluation_time_seconds": result.evaluation_time_seconds,
            "harness_results": harness_elos,
        }

    async def _run_nnue_baseline_evaluation(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        recording_config: Any = None,
    ) -> dict[str, Any] | None:
        """Run evaluation against NNUE-based baselines (Dec 31, 2025).

        Tests the model against NNUE baselines to compare NN vs NNUE performance:
        - 2-player: Uses NNUE_MINIMAX_D4
        - 3-4 player: Uses NNUE_BRS_D3 and NNUE_MAXN_D3

        Returns:
            Results dict with win rates per NNUE baseline, or None if NNUE unavailable
        """
        from pathlib import Path

        # Check if NNUE model exists for this config
        nnue_model_path = Path(f"models/nnue/nnue_{board_type}_{num_players}p.pt")
        if not nnue_model_path.exists():
            # Also check alternative naming convention
            nnue_model_path = Path(f"models/nnue_{board_type}_{num_players}p.pt")
            if not nnue_model_path.exists():
                logger.debug(f"No NNUE model found for {board_type}_{num_players}p")
                return None

        try:
            from app.training.game_gauntlet import BaselineOpponent, run_baseline_gauntlet

            # Select appropriate NNUE baselines based on player count
            if num_players == 2:
                nnue_opponents = [BaselineOpponent.NNUE_MINIMAX_D4]
            else:
                nnue_opponents = [
                    BaselineOpponent.NNUE_BRS_D3,
                    BaselineOpponent.NNUE_MAXN_D3,
                ]

            # Run gauntlet against NNUE baselines
            gauntlet_results = await asyncio.wait_for(
                asyncio.to_thread(
                    run_baseline_gauntlet,
                    model_path=model_path,
                    board_type=board_type,
                    num_players=num_players,
                    games_per_opponent=self.config.games_per_baseline,
                    opponents=nnue_opponents,
                    recording_config=recording_config,
                ),
                timeout=self.config.evaluation_timeout_seconds / 2,  # Use half timeout
            )

            # Extract win rates per NNUE opponent
            results = {
                "nnue_model": str(nnue_model_path),
                "win_rates": {},
            }
            for opponent, stats in gauntlet_results.opponent_results.items():
                results["win_rates"][opponent] = stats.get("win_rate", 0.0)

            return results

        except asyncio.TimeoutError:
            logger.warning(f"NNUE baseline evaluation timed out for {board_type}_{num_players}p")
            return None
        except ImportError as e:
            logger.debug(f"NNUE baseline dependencies not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"NNUE baseline evaluation error: {e}")
            return None

    async def _emit_harness_evaluation_completed(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        harness_type: str,
        elo: float,
        win_rate: float,
    ) -> None:
        """Emit HARNESS_EVALUATION_COMPLETED event for per-harness Elo tracking.

        December 31, 2025: Enables tracking Elo per (model, harness) combination.
        """
        config_key = make_config_key(board_type, num_players)
        safe_emit_event(
            "harness_evaluation_completed",
            {
                "config_key": config_key,
                "model_path": model_path,
                "harness_type": harness_type,
                "elo": elo,
                "win_rate": win_rate,
                "source": "tournament_daemon",
            },
            context="Tournament",
        )

    async def _run_ladder_tournament(self) -> dict[str, Any]:
        """Run a ladder tournament across all configurations.

        Returns:
            Tournament results dict
        """
        self._tournament_stats.evaluations_completed += 1  # Fixed: use base class field, not property
        start_time = time.time()

        results = {
            "tournament_id": str(uuid.uuid4()),
            "success": False,
            "configs_evaluated": 0,
        }

        try:
            # Find all tournament models (includes both canonical and ringrift_best_*)
            from app.models.discovery import find_tournament_models

            models = find_tournament_models()

            # find_tournament_models() returns {(board_type, num_players): Path}
            for (board_type, num_players), model_path in models.items():
                # Queue evaluation
                self._evaluation_queue.put_nowait({
                    "model_path": str(model_path),
                    "board_type": board_type,
                    "num_players": num_players,
                    "trigger": "periodic_ladder",
                })

                results["configs_evaluated"] += 1

            results["success"] = True
            self._tournament_stats.last_tournament_time = time.time()

        except ImportError:
            logger.warning("Model discovery not available, skipping ladder tournament")
        except Exception as e:
            logger.error(f"Ladder tournament failed: {e}")
            results["error"] = str(e)
            self._tournament_stats.errors.append(str(e))

        results["duration_seconds"] = time.time() - start_time
        return results

    async def _run_calibration_tournament(self) -> dict[str, Any]:
        """Run calibration tournament to validate Elo ladder.

        Tests expected win rates between baseline opponents:
        - Heuristic vs Random: ~95%+ win rate (validates 800 Elo gap)
        - Heuristic_Strong vs Heuristic: ~60% win rate (validates 200 Elo gap)
        - MCTS_Light vs Heuristic_Strong: ~55% win rate (validates 100 Elo gap)

        Returns:
            Tournament results with calibration validation status
        """
        logger.info("Running calibration tournament to validate Elo ladder")
        start_time = time.time()

        results = {
            "tournament_id": str(uuid.uuid4()),
            "tournament_type": "calibration",
            "success": False,
            "matchups": {},
        }

        try:
            from app.training.game_gauntlet import (
                create_baseline_ai,
                play_single_game,
                BaselineOpponent,
            )
            from app.models import BoardType

            # Define calibration matchups: (stronger, weaker, expected_win_rate)
            # Dec 31, 2025: Added NNUE calibration pairs for architecture diversity
            calibration_pairs = [
                (BaselineOpponent.HEURISTIC, BaselineOpponent.RANDOM, 0.90),
                (BaselineOpponent.HEURISTIC_STRONG, BaselineOpponent.HEURISTIC, 0.55),
                (BaselineOpponent.MCTS_LIGHT, BaselineOpponent.HEURISTIC_STRONG, 0.55),
                # NNUE calibration (NNUE_MINIMAX should beat HEURISTIC_STRONG)
                (BaselineOpponent.NNUE_MINIMAX_D4, BaselineOpponent.HEURISTIC_STRONG, 0.55),
            ]

            # Use square8_2p for calibration (fast games)
            board_type = BoardType.SQUARE8
            num_players = 2
            games_per_matchup = self.config.calibration_games

            # Create recording config if enabled (Dec 2025 - tournament games for training)
            recording_config = None
            if self.config.enable_game_recording:
                try:
                    from app.db.unified_recording import RecordingConfig, RecordSource
                    recording_config = RecordingConfig(
                        board_type=board_type.value,
                        num_players=num_players,
                        source=RecordSource.TOURNAMENT,
                        engine_mode="calibration",
                        db_prefix=self.config.recording_db_prefix,
                        db_dir=self.config.recording_db_dir,
                        store_history_entries=True,
                    )
                except ImportError:
                    logger.debug("Recording module not available for calibration tournament")

            for stronger, weaker, expected_rate in calibration_pairs:
                matchup_key = f"{stronger.value}_vs_{weaker.value}"
                wins = 0

                for game_num in range(games_per_matchup):
                    try:
                        # Alternate which player is "stronger" for position fairness
                        # Dec 29: Fixed argument order (baseline, player, board_type, num_players)
                        if game_num % 2 == 0:
                            player_0_ai = create_baseline_ai(
                                stronger, 1, board_type, num_players=num_players
                            )
                            player_1_ai = create_baseline_ai(
                                weaker, 2, board_type, num_players=num_players
                            )
                            stronger_player = 0
                        else:
                            player_0_ai = create_baseline_ai(
                                weaker, 1, board_type, num_players=num_players
                            )
                            player_1_ai = create_baseline_ai(
                                stronger, 2, board_type, num_players=num_players
                            )

                        game_result = play_single_game(
                            board_type=board_type,
                            num_players=num_players,
                            player_ais=[player_0_ai, player_1_ai],
                            timeout=self.config.game_timeout_seconds,
                            recording_config=recording_config,
                        )

                        if game_result.get("winner") == stronger_player:
                            wins += 1

                        self._tournament_stats.games_played += 1

                    except Exception as e:
                        logger.warning(f"Calibration game failed: {e}")

                actual_rate = wins / games_per_matchup if games_per_matchup > 0 else 0
                # Allow 10% margin below expected rate
                calibration_valid = actual_rate >= expected_rate * 0.9

                results["matchups"][matchup_key] = {
                    "wins": wins,
                    "games": games_per_matchup,
                    "win_rate": actual_rate,
                    "expected_rate": expected_rate,
                    "calibration_valid": calibration_valid,
                }

                if not calibration_valid:
                    logger.warning(
                        f"Calibration FAILED: {matchup_key} win rate {actual_rate:.1%} "
                        f"below expected {expected_rate:.1%}"
                    )

            results["success"] = True
            results["all_valid"] = all(
                m["calibration_valid"] for m in results["matchups"].values()
            )

        except ImportError as e:
            logger.warning(f"Calibration tournament dependencies not available: {e}")
            results["error"] = str(e)
        except Exception as e:
            logger.error(f"Calibration tournament failed: {e}")
            results["error"] = str(e)
            self._tournament_stats.errors.append(str(e))

        results["duration_seconds"] = time.time() - start_time
        logger.info(f"Calibration tournament completed: {results.get('all_valid', False)}")
        return results

    async def _run_cross_nn_tournament(self) -> dict[str, Any]:
        """Run cross-NN version tournament to compare model generations (Dec 2025).

        Discovers all model versions for each configuration and runs tournaments
        between adjacent versions (e.g., v2 vs v3, v3 vs v4) to:
        - Validate newer models are stronger than older ones
        - Maintain accurate Elo ratings across model generations
        - Identify potential regressions in model quality

        Returns:
            Tournament results with per-pairing win rates and Elo updates
        """
        logger.info("Running cross-NN version tournament")
        start_time = time.time()

        results = {
            "tournament_id": str(uuid.uuid4()),
            "tournament_type": "cross_nn",
            "success": False,
            "pairings": {},
            "games_played": 0,
        }

        try:
            from app.training.game_gauntlet import play_single_game
            from app.models import BoardType
            from app.ai.neural_net import UnifiedNeuralNetFactory
            from app.training.elo_service import get_elo_service
            from pathlib import Path
            import re

            elo_service = get_elo_service()
            models_dir = Path("models")

            # Find all canonical models per config
            # Pattern: canonical_{board}_{n}p.pth or canonical_{board}_{n}p_v{version}.pth
            model_pattern = re.compile(
                r"canonical_(?P<board>\w+)_(?P<players>\d)p(?:_v(?P<version>\d+))?\.pth"
            )

            # Group models by config (board_type, num_players)
            config_models: dict[tuple[str, int], list[tuple[str, Path]]] = {}

            for model_path in models_dir.glob("canonical_*.pth"):
                match = model_pattern.match(model_path.name)
                if match:
                    board = match.group("board")
                    players = int(match.group("players"))
                    version = match.group("version") or "base"
                    config_key = (board, players)

                    if config_key not in config_models:
                        config_models[config_key] = []
                    config_models[config_key].append((version, model_path))

            # Also check for versioned models like hex8_2p_v2.pth, hex8_2p_v3.pth
            version_pattern = re.compile(
                r"(?:canonical_)?(?P<board>\w+)_(?P<players>\d)p_v(?P<version>\d+)\.pth"
            )

            for model_path in models_dir.glob("*_v*.pth"):
                if "canonical" in model_path.name:
                    continue  # Already captured above
                match = version_pattern.match(model_path.name)
                if match:
                    board = match.group("board")
                    players = int(match.group("players"))
                    version = f"v{match.group('version')}"
                    config_key = (board, players)

                    if config_key not in config_models:
                        config_models[config_key] = []
                    config_models[config_key].append((version, model_path))

            # Dec 31, 2025: Also discover named architecture variants (v5heavy, v5-heavy-large)
            # Pattern: canonical_{board}_{n}p_{variant}.pth
            variant_pattern = re.compile(
                r"canonical_(?P<board>\w+)_(?P<players>\d)p_(?P<variant>v5heavy|v5-heavy|v5-heavy-large|v4|nnue)\.pth"
            )

            for model_path in models_dir.glob("canonical_*_*.pth"):
                match = variant_pattern.match(model_path.name)
                if match:
                    board = match.group("board")
                    players = int(match.group("players"))
                    variant = match.group("variant")
                    config_key = (board, players)

                    if config_key not in config_models:
                        config_models[config_key] = []
                    # Use variant name as version identifier
                    config_models[config_key].append((variant, model_path))

            games_per_pairing = self.config.cross_nn_games_per_pairing
            total_games = 0

            for (board, num_players), models in config_models.items():
                if len(models) < 2:
                    continue  # Need at least 2 versions to compare

                # Sort by version (base < v2 < v3 < ... < v4 < v5heavy < nnue)
                # Dec 31, 2025: Extended to handle named architecture variants
                def version_key(item: tuple[str, Path]) -> tuple[int, str]:
                    v = item[0]
                    # Known architectures in order of complexity/recency
                    version_order = {
                        "base": (0, ""),
                        "v2": (2, ""),
                        "v3": (3, ""),
                        "v4": (4, ""),
                        "v5heavy": (5, "heavy"),
                        "v5-heavy": (5, "heavy"),
                        "v5-heavy-large": (5, "heavy-large"),
                        "nnue": (6, ""),  # NNUE is evaluated separately
                    }
                    if v in version_order:
                        return version_order[v]
                    # Handle numeric versions like "v5", "v10"
                    if v.startswith("v") and v[1:].isdigit():
                        return (int(v[1:]), "")
                    return (100, v)  # Unknown versions sort last

                models.sort(key=version_key)

                # Get board type enum
                try:
                    board_type = BoardType(board)
                except ValueError:
                    logger.warning(f"Unknown board type: {board}")
                    continue

                # Create recording config if enabled (Dec 2025 - tournament games for training)
                recording_config = None
                if self.config.enable_game_recording:
                    try:
                        from app.db.unified_recording import RecordingConfig, RecordSource
                        recording_config = RecordingConfig(
                            board_type=board_type.value,
                            num_players=num_players,
                            source=RecordSource.TOURNAMENT,
                            engine_mode="cross_nn",
                            db_prefix=self.config.recording_db_prefix,
                            db_dir=self.config.recording_db_dir,
                            store_history_entries=True,
                        )
                    except ImportError:
                        logger.debug("Recording module not available for cross-NN tournament")

                # Run tournaments between adjacent versions
                for i in range(len(models) - 1):
                    older_version, older_path = models[i]
                    newer_version, newer_path = models[i + 1]

                    pairing_key = f"{board}_{num_players}p:{older_version}_vs_{newer_version}"
                    logger.info(f"Cross-NN pairing: {pairing_key}")

                    # Load models
                    try:
                        older_ai = UnifiedNeuralNetFactory.create(
                            str(older_path),
                            board_type=board_type,
                            num_players=num_players,
                        )
                        newer_ai = UnifiedNeuralNetFactory.create(
                            str(newer_path),
                            board_type=board_type,
                            num_players=num_players,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load models for {pairing_key}: {e}")
                        results["pairings"][pairing_key] = {"error": str(e)}
                        continue

                    wins_newer = 0
                    wins_older = 0

                    for game_num in range(games_per_pairing):
                        try:
                            # Alternate positions for fairness
                            if game_num % 2 == 0:
                                player_ais = [newer_ai, older_ai]
                                newer_player = 0
                            else:
                                player_ais = [older_ai, newer_ai]
                                newer_player = 1

                            game_result = play_single_game(
                                board_type=board_type,
                                num_players=num_players,
                                player_ais=player_ais,
                                timeout=self.config.game_timeout_seconds,
                                recording_config=recording_config,
                            )

                            winner = game_result.get("winner")
                            if winner == newer_player:
                                wins_newer += 1
                            elif winner is not None:
                                wins_older += 1

                            total_games += 1
                            self._tournament_stats.games_played += 1

                            # Record match for Elo update
                            if winner is not None:
                                # Jan 2026: Fixed incorrect parameter names (was winner_id/loser_id)
                                winner_model_id = newer_path.stem if winner == newer_player else older_path.stem
                                loser_model_id = older_path.stem if winner == newer_player else newer_path.stem
                                # January 2026: Extract harness_type for per-harness Elo tracking
                                # Default to gumbel_mcts for legacy model names without composite ID
                                harness_type = extract_harness_type(winner_model_id) or "gumbel_mcts"
                                elo_service.record_match(
                                    participant_a=winner_model_id,
                                    participant_b=loser_model_id,
                                    winner=winner_model_id,
                                    board_type=board,
                                    num_players=num_players,
                                    harness_type=harness_type,
                                )

                        except Exception as e:
                            logger.warning(f"Cross-NN game failed: {e}")

                    win_rate_newer = wins_newer / games_per_pairing if games_per_pairing > 0 else 0
                    # Newer model should win >50% if it's actually better
                    improvement_validated = win_rate_newer >= 0.5

                    results["pairings"][pairing_key] = {
                        "newer_wins": wins_newer,
                        "older_wins": wins_older,
                        "draws": games_per_pairing - wins_newer - wins_older,
                        "games": games_per_pairing,
                        "newer_win_rate": win_rate_newer,
                        "improvement_validated": improvement_validated,
                    }

                    if not improvement_validated:
                        logger.warning(
                            f"Potential regression: {newer_version} only {win_rate_newer:.1%} "
                            f"vs {older_version} in {board}_{num_players}p"
                        )

            results["success"] = True
            results["games_played"] = total_games
            results["configs_tested"] = len([k for k, v in config_models.items() if len(v) >= 2])

        except ImportError as e:
            logger.warning(f"Cross-NN tournament dependencies not available: {e}")
            results["error"] = str(e)
        except Exception as e:
            logger.error(f"Cross-NN tournament failed: {e}")
            results["error"] = str(e)
            self._tournament_stats.errors.append(str(e))

        results["duration_seconds"] = time.time() - start_time
        logger.info(f"Cross-NN tournament completed: {results.get('games_played', 0)} games")
        return results

    async def _update_elo(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        gauntlet_results: Any,
    ) -> None:
        """Update ELO ratings based on gauntlet results.

        NOTE (December 2025): Match recording is now done inline in
        game_gauntlet._evaluate_single_opponent(). This method now only
        ensures model registration - individual matches are NOT re-recorded
        to avoid double-counting Elo changes.
        """
        try:
            from app.training.elo_service import get_elo_service

            elo_service = get_elo_service()
            model_id = Path(model_path).stem

            # Register model if not already registered
            elo_service.register_model(
                model_id=model_id,
                board_type=board_type,
                num_players=num_players,
                model_path=model_path,
            )

            # December 2025: Match recording moved to game_gauntlet.py
            # Games are now recorded inline during gauntlet evaluation
            # to ensure ALL configs (including 3p/4p) are tracked.
            # Previously, only games through tournament_daemon were recorded.
            if hasattr(gauntlet_results, "opponent_results"):
                total_games = sum(
                    int(stats.get("games", 0))
                    for stats in gauntlet_results.opponent_results.values()
                )
                logger.info(
                    f"ELO for {model_id}: {total_games} games already recorded inline "
                    f"(estimated_elo={gauntlet_results.estimated_elo:.0f})"
                )
                return

            # Legacy path for older gauntlet result formats (now deprecated)
            match_results = gauntlet_results.get("matches", [])
            for match in match_results:
                opponent_id = match.get("opponent")
                winner = match.get("winner")

                if not opponent_id:
                    continue

                winner_id = model_id if winner == 0 else (opponent_id if winner == 1 else None)

                # January 2026: Extract harness_type for per-harness Elo tracking
                # Default to gumbel_mcts for legacy model names without composite ID
                harness_type = extract_harness_type(model_id) or "gumbel_mcts"
                elo_service.record_match(
                    participant_a=model_id,
                    participant_b=opponent_id,
                    winner=winner_id,
                    board_type=board_type,
                    num_players=num_players,
                    game_length=match.get("game_length", 0),
                    duration_sec=match.get("duration", 0.0),
                    harness_type=harness_type,
                )

            logger.info(f"Updated ELO for {model_id} with {len(match_results)} matches")

        except ImportError:
            logger.warning("EloService not available for ELO updates")
        except Exception as e:
            logger.error(f"Failed to update ELO: {e}")

    async def _emit_evaluation_completed(self, results: dict[str, Any]) -> None:
        """Emit EVALUATION_COMPLETED event."""
        try:
            from app.coordination.event_router import publish, DataEventType

            await publish(
                event_type=DataEventType.EVALUATION_COMPLETED,
                payload={
                    "model_path": results.get("model_path"),
                    "board_type": results.get("board_type"),
                    "num_players": results.get("num_players"),
                    "success": results.get("success", False),
                    "win_rates": results.get("win_rates", {}),
                    "elo": results.get("elo"),
                    "games_played": results.get("games_played", 0),
                },
                source="tournament_daemon",
            )

        except ImportError:
            pass  # Event router not available
        except Exception as e:
            logger.debug(f"Failed to emit EVALUATION_COMPLETED: {e}")

    async def evaluate_model(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
    ) -> dict[str, Any]:
        """Manually trigger evaluation for a model.

        Args:
            model_path: Path to model checkpoint
            board_type: Board type
            num_players: Number of players

        Returns:
            Evaluation results
        """
        return await self._run_evaluation(
            model_path=model_path,
            board_type=board_type,
            num_players=num_players,
            trigger="manual",
        )

    # =========================================================================
    # Jan 2026: Periodic Multi-Harness, Stale Evaluation, and Cross-Config Loops
    # =========================================================================

    async def _periodic_multi_harness_loop(self) -> None:
        """Periodic multi-harness evaluation loop (Jan 2026).

        Runs multi-harness evaluation on top models at regular intervals
        to ensure harness-specific Elo ratings stay fresh.
        """
        # Initial delay to stagger with other loops
        await asyncio.sleep(300)  # 5 minute initial delay

        while self._running:
            try:
                await asyncio.sleep(self.config.periodic_multi_harness_interval_seconds)

                if not self._running:
                    break

                logger.info("Running periodic multi-harness evaluation")
                results = await self._run_periodic_multi_harness()

                if results.get("success"):
                    configs_evaluated = results.get("configs_evaluated", 0)
                    total_games = results.get("total_games", 0)
                    logger.info(
                        f"Periodic multi-harness complete: {configs_evaluated} configs, "
                        f"{total_games} games"
                    )
                else:
                    logger.warning(
                        f"Periodic multi-harness failed: {results.get('error')}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic multi-harness loop: {e}")
                self._tournament_stats.record_failure(str(e))

    async def _run_periodic_multi_harness(self) -> dict[str, Any]:
        """Run multi-harness evaluation on all priority configs.

        Returns:
            Results dict with configs evaluated and games played
        """
        results = {
            "success": False,
            "configs_evaluated": 0,
            "total_games": 0,
            "harness_results": {},
        }

        try:
            from app.models.discovery import find_tournament_models

            models = find_tournament_models()
            priority_configs = set(self.config.multi_harness_priority_configs)

            for (board_type, num_players), model_path in models.items():
                config_key = make_config_key(board_type, num_players)

                # Prioritize configured configs, but still run others
                # Feb 2026: Raised cap from 4 to 12 to evaluate all configs
                if config_key not in priority_configs and results["configs_evaluated"] >= 12:
                    continue  # Limit non-priority configs per cycle

                try:
                    harness_results = await self._run_multi_harness_evaluation(
                        str(model_path), board_type, num_players
                    )

                    results["configs_evaluated"] += 1
                    results["total_games"] += harness_results.get("total_games", 0)
                    results["harness_results"][config_key] = harness_results

                    # Update last evaluation time
                    self._last_harness_eval_time[config_key] = time.time()

                    # Emit event for feedback to selfplay scheduler
                    try:
                        from app.distributed.data_events import DataEventType
                        safe_emit_event(
                            DataEventType.MULTI_HARNESS_EVALUATION_COMPLETED.value,
                            {
                                "config_key": config_key,
                                "best_harness": harness_results.get("best_harness"),
                                "best_elo": harness_results.get("best_elo"),
                                "harness_results": harness_results.get("harness_results", {}),
                            },
                            context="TournamentDaemon",
                        )
                    except ImportError:
                        pass  # Event emission not critical

                except Exception as e:
                    logger.warning(f"Multi-harness eval failed for {config_key}: {e}")

            results["success"] = True

        except ImportError:
            logger.warning("Model discovery not available for multi-harness")
            results["error"] = "import_error"
        except Exception as e:
            logger.error(f"Periodic multi-harness failed: {e}")
            results["error"] = str(e)

        return results

    async def _stale_evaluation_loop(self) -> None:
        """Stale evaluation detection loop (Jan 2026).

        Checks for models with stale ratings and queues re-evaluation.
        """
        # Initial delay
        await asyncio.sleep(600)  # 10 minute initial delay

        while self._running:
            try:
                await asyncio.sleep(self.config.stale_evaluation_check_interval_seconds)

                if not self._running:
                    break

                logger.debug("Checking for stale evaluations")
                stale_models = await self._find_stale_models()

                if stale_models:
                    logger.info(f"Found {len(stale_models)} models with stale evaluations")
                    for model_info in stale_models:
                        self._evaluation_queue.put_nowait({
                            "model_path": model_info["model_path"],
                            "board_type": model_info["board_type"],
                            "num_players": model_info["num_players"],
                            "trigger": "stale_evaluation",
                        })

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stale evaluation loop: {e}")

    async def _find_stale_models(self) -> list[dict[str, Any]]:
        """Find models with stale Elo ratings.

        Returns:
            List of model info dicts for models needing re-evaluation
        """
        stale_models = []
        threshold_seconds = self.config.stale_evaluation_threshold_days * 24 * 3600
        current_time = time.time()

        try:
            from app.models.discovery import find_tournament_models
            from app.training.elo_service import get_elo_service

            models = find_tournament_models()
            elo_service = get_elo_service()

            for (board_type, num_players), model_path in models.items():
                model_id = Path(model_path).stem
                config_key = make_config_key(board_type, num_players)

                # Check when model was last evaluated
                last_eval = self._last_harness_eval_time.get(config_key, 0)

                # Also check Elo service for last match time
                try:
                    rating_info = elo_service.get_rating(model_id)
                    if rating_info and hasattr(rating_info, "last_match_time"):
                        last_match = rating_info.last_match_time or 0
                        last_eval = max(last_eval, last_match)
                except Exception:
                    pass

                # Check if stale
                if current_time - last_eval > threshold_seconds:
                    stale_models.append({
                        "model_path": str(model_path),
                        "board_type": board_type,
                        "num_players": num_players,
                        "config_key": config_key,
                        "days_stale": (current_time - last_eval) / (24 * 3600),
                    })

        except ImportError:
            logger.debug("Dependencies not available for stale detection")
        except Exception as e:
            logger.warning(f"Error finding stale models: {e}")

        return stale_models

    async def _cross_config_loop(self) -> None:
        """Cross-config tournament loop (Jan 2026).

        Runs tournaments between models of different configurations
        within the same board family to validate transfer learning.
        """
        # Initial delay
        await asyncio.sleep(900)  # 15 minute initial delay

        while self._running:
            try:
                await asyncio.sleep(self.config.cross_config_interval_seconds)

                if not self._running:
                    break

                logger.info("Running cross-config tournament")
                results = await self._run_cross_config_tournament()

                if results.get("success"):
                    families_evaluated = results.get("families_evaluated", 0)
                    total_games = results.get("total_games", 0)
                    logger.info(
                        f"Cross-config tournament complete: {families_evaluated} families, "
                        f"{total_games} games"
                    )
                else:
                    logger.warning(
                        f"Cross-config tournament failed: {results.get('error')}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cross-config loop: {e}")
                self._tournament_stats.record_failure(str(e))

    async def _run_cross_config_tournament(self) -> dict[str, Any]:
        """Run cross-config tournament within board families.

        Compares models trained on different player counts (2p vs 3p vs 4p)
        within the same board type to validate curriculum progression.

        Returns:
            Results dict with families evaluated and games played
        """
        results = {
            "success": False,
            "families_evaluated": 0,
            "total_games": 0,
            "family_results": {},
        }

        try:
            from app.models.discovery import find_tournament_models
            from app.training.game_gauntlet import play_single_game
            from app.training.elo_service import get_elo_service
            from app.models import BoardType

            models = find_tournament_models()
            elo_service = get_elo_service()
            games_per_matchup = self.config.cross_config_games_per_matchup

            for family in self.config.cross_config_families:
                family_models = []

                # Collect models for this family
                for config_key in family:
                    parsed = parse_config_key(config_key)
                    if not parsed:
                        continue

                    model_key = (parsed.board_type, parsed.num_players)
                    if model_key in models:
                        family_models.append({
                            "config_key": config_key,
                            "board_type": parsed.board_type,
                            "num_players": parsed.num_players,
                            "model_path": models[model_key],
                        })

                if len(family_models) < 2:
                    continue  # Need at least 2 models to compare

                family_key = "_".join(m["config_key"] for m in family_models)
                family_result = {
                    "matchups": {},
                    "games_played": 0,
                }

                # Run round-robin within family (comparing different player counts)
                # Note: This is primarily for transfer learning validation
                # We compare how 2p model performs when adapted to 4p scenarios, etc.
                for i, model_a in enumerate(family_models):
                    for model_b in family_models[i + 1:]:
                        matchup_key = f"{model_a['config_key']}_vs_{model_b['config_key']}"

                        # Use the larger player count for the matchup
                        num_players = max(model_a["num_players"], model_b["num_players"])
                        board_type = model_a["board_type"]

                        wins_a = 0
                        for game_num in range(games_per_matchup):
                            try:
                                # Alternate starting positions
                                if game_num % 2 == 0:
                                    player_models = [model_a["model_path"], model_b["model_path"]]
                                    model_a_player = 0
                                else:
                                    player_models = [model_b["model_path"], model_a["model_path"]]
                                    model_a_player = 1

                                # Create player AIs for additional players if needed
                                from app.training.game_gauntlet import (
                                    create_neural_ai,
                                    BaselineOpponent,
                                    create_baseline_ai,
                                )

                                player_ais = []
                                for p in range(num_players):
                                    if p < 2:
                                        # Use the competing models
                                        player_ais.append(
                                            create_neural_ai(
                                                str(player_models[p]),
                                                p + 1,
                                                BoardType(board_type),
                                                num_players,
                                            )
                                        )
                                    else:
                                        # Fill with heuristic for 3p/4p games
                                        player_ais.append(
                                            create_baseline_ai(
                                                BaselineOpponent.HEURISTIC,
                                                p + 1,
                                                BoardType(board_type),
                                                num_players,
                                            )
                                        )

                                game_result = play_single_game(
                                    board_type=BoardType(board_type),
                                    num_players=num_players,
                                    player_ais=player_ais,
                                    timeout=self.config.game_timeout_seconds,
                                )

                                winner = game_result.get("winner")
                                if winner == model_a_player:
                                    wins_a += 1

                                family_result["games_played"] += 1
                                results["total_games"] += 1
                                self._tournament_stats.games_played += 1

                            except Exception as e:
                                logger.warning(f"Cross-config game failed: {e}")

                        win_rate_a = wins_a / games_per_matchup if games_per_matchup > 0 else 0
                        family_result["matchups"][matchup_key] = {
                            "model_a": model_a["config_key"],
                            "model_b": model_b["config_key"],
                            "wins_a": wins_a,
                            "games": games_per_matchup,
                            "win_rate_a": win_rate_a,
                        }

                        # Feb 2026: Do NOT record cross-config matches in the main
                        # Elo database. Recording a 2p model as a participant in 4p
                        # games pollutes per-config Elo tracking and causes massive
                        # Elo regression (e.g., hex8_4p dropped from 1900 to 1508).
                        # Cross-config results are logged above in family_result for
                        # informational purposes only.
                        logger.info(
                            f"Cross-config: {model_a['config_key']} vs {model_b['config_key']} "
                            f"({board_type} {num_players}p) - win_rate_a={win_rate_a:.1%}"
                        )

                results["family_results"][family_key] = family_result
                results["families_evaluated"] += 1

            results["success"] = True

            # Emit event for cross-config results
            try:
                from app.distributed.data_events import DataEventType
                safe_emit_event(
                    DataEventType.CROSS_CONFIG_TOURNAMENT_COMPLETED.value,
                    {
                        "families_evaluated": results["families_evaluated"],
                        "total_games": results["total_games"],
                        "family_results": results["family_results"],
                    },
                    context="TournamentDaemon",
                )
            except ImportError:
                pass  # Event emission not critical

        except ImportError as e:
            logger.warning(f"Cross-config tournament dependencies not available: {e}")
            results["error"] = "import_error"
        except Exception as e:
            logger.error(f"Cross-config tournament failed: {e}")
            results["error"] = str(e)

        return results

    async def _periodic_topn_roundrobin_loop(self) -> None:
        """Top-N round-robin tournament loop (Jan 2026).

        Periodically runs round-robin tournaments between top-rated models
        for each configuration. This generates high-quality training data
        from diverse model matchups.
        """
        # Initial delay to let cluster stabilize
        await asyncio.sleep(600)  # 10 minute initial delay

        while self._running:
            try:
                await asyncio.sleep(self.config.topn_roundrobin_interval_seconds)

                if not self._running:
                    break

                logger.info("Running top-N round-robin tournaments")
                results = await self._run_topn_roundrobin_tournament()

                if results.get("success"):
                    configs_evaluated = results.get("configs_evaluated", 0)
                    total_games = results.get("total_games", 0)
                    logger.info(
                        f"Top-N round-robin complete: {configs_evaluated} configs, "
                        f"{total_games} games played"
                    )
                else:
                    logger.warning(
                        f"Top-N round-robin failed: {results.get('error')}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in top-N round-robin loop: {e}")
                self._tournament_stats.record_failure(str(e))

    async def _run_topn_roundrobin_tournament(self) -> dict[str, Any]:
        """Run round-robin tournaments between top-rated models.

        For each configured board/player configuration, gets the top N
        models by Elo rating and runs a round-robin tournament where
        each model plays every other model.

        Returns:
            Results dict with configs evaluated, games played, and per-config results
        """
        results = {
            "success": False,
            "configs_evaluated": 0,
            "total_games": 0,
            "config_results": {},
        }

        try:
            from app.training.elo_service import get_elo_service
            from app.models.discovery import find_tournament_models
            from app.training.game_gauntlet import (
                play_single_game,
                create_neural_ai,
            )
            from app.models import BoardType
            from app.training.composite_participant import extract_harness_type

            elo_service = get_elo_service()
            available_models = find_tournament_models()
            games_per_matchup = self.config.topn_roundrobin_games_per_matchup
            top_n = self.config.topn_roundrobin_n
            min_games = self.config.topn_roundrobin_min_elo_games

            # Get configs to evaluate
            configs_to_evaluate = self.config.topn_roundrobin_configs
            if not configs_to_evaluate:
                # Use all available configs
                configs_to_evaluate = [
                    make_config_key(bt, np)
                    for (bt, np) in available_models.keys()
                ]

            for config_key in configs_to_evaluate:
                parsed = parse_config_key(config_key)
                if not parsed:
                    logger.warning(f"Invalid config key: {config_key}")
                    continue

                board_type = parsed.board_type
                num_players = parsed.num_players

                # Get top N models for this config
                try:
                    leaderboard = elo_service.get_leaderboard(
                        board_type=board_type,
                        num_players=num_players,
                        top_n=top_n,
                        min_games=min_games,
                    )
                except Exception as e:
                    logger.warning(f"Failed to get leaderboard for {config_key}: {e}")
                    continue

                if len(leaderboard) < 2:
                    logger.debug(f"Not enough rated models for {config_key} round-robin")
                    continue

                config_result = {
                    "models": len(leaderboard),
                    "matchups": {},
                    "games_played": 0,
                }

                # Extract model paths from leaderboard entries
                top_models = []
                for entry in leaderboard:
                    participant_id = entry.get("participant_id", entry.get("participant"))
                    if not participant_id:
                        continue
                    # Find model path - try to match with available models
                    model_key = (board_type, num_players)
                    if model_key in available_models:
                        # Use canonical model path
                        model_path = available_models[model_key]
                    else:
                        # Try to construct path from participant_id
                        model_path = Path(f"models/{participant_id}.pth")
                        if not model_path.exists():
                            model_path = Path(f"models/canonical_{config_key}.pth")
                            if not model_path.exists():
                                continue

                    top_models.append({
                        "participant_id": participant_id,
                        "model_path": str(model_path),
                        "elo": entry.get("elo", entry.get("rating", 1500)),
                    })

                if len(top_models) < 2:
                    continue

                # Run round-robin tournament
                for i, model_a in enumerate(top_models):
                    for model_b in top_models[i + 1:]:
                        matchup_key = f"{model_a['participant_id']}_vs_{model_b['participant_id']}"
                        wins_a = 0

                        for game_num in range(games_per_matchup):
                            try:
                                # Alternate starting positions
                                if game_num % 2 == 0:
                                    path_a, path_b = model_a["model_path"], model_b["model_path"]
                                    model_a_player = 0
                                else:
                                    path_a, path_b = model_b["model_path"], model_a["model_path"]
                                    model_a_player = 1

                                # Create player AIs
                                player_ais = []
                                player_ais.append(create_neural_ai(
                                    path_a, 1, BoardType(board_type), num_players
                                ))
                                player_ais.append(create_neural_ai(
                                    path_b, 2, BoardType(board_type), num_players
                                ))

                                # Fill remaining slots with the top model
                                for p in range(2, num_players):
                                    player_ais.append(create_neural_ai(
                                        model_a["model_path"], p + 1,
                                        BoardType(board_type), num_players
                                    ))

                                game_result = play_single_game(
                                    board_type=BoardType(board_type),
                                    num_players=num_players,
                                    player_ais=player_ais,
                                    timeout=self.config.game_timeout_seconds,
                                )

                                winner = game_result.get("winner")
                                if winner == model_a_player:
                                    wins_a += 1

                                config_result["games_played"] += 1
                                results["total_games"] += 1
                                self._tournament_stats.games_played += 1

                            except Exception as e:
                                logger.warning(f"Top-N game failed: {e}")

                        # Record matchup results
                        win_rate_a = wins_a / games_per_matchup if games_per_matchup > 0 else 0
                        config_result["matchups"][matchup_key] = {
                            "model_a": model_a["participant_id"],
                            "model_b": model_b["participant_id"],
                            "wins_a": wins_a,
                            "games": games_per_matchup,
                            "win_rate_a": win_rate_a,
                        }

                        # Record in Elo service
                        try:
                            # January 2026: Default to gumbel_mcts for legacy model names
                            harness_type = extract_harness_type(model_a["participant_id"]) or "gumbel_mcts"
                            for _ in range(wins_a):
                                elo_service.record_match(
                                    participant_a=model_a["participant_id"],
                                    participant_b=model_b["participant_id"],
                                    winner=model_a["participant_id"],
                                    board_type=board_type,
                                    num_players=num_players,
                                    tournament_id=f"topn_roundrobin_{config_key}",
                                    harness_type=harness_type,
                                )
                            for _ in range(games_per_matchup - wins_a):
                                elo_service.record_match(
                                    participant_a=model_a["participant_id"],
                                    participant_b=model_b["participant_id"],
                                    winner=model_b["participant_id"],
                                    board_type=board_type,
                                    num_players=num_players,
                                    tournament_id=f"topn_roundrobin_{config_key}",
                                    harness_type=harness_type,
                                )
                        except Exception as e:
                            logger.warning(f"Failed to record top-N Elo: {e}")

                results["config_results"][config_key] = config_result
                results["configs_evaluated"] += 1

            results["success"] = True

            # Emit event for top-N round-robin results
            try:
                from app.distributed.data_events import DataEventType
                safe_emit_event(
                    DataEventType.TOPN_ROUNDROBIN_COMPLETED.value,
                    {
                        "configs_evaluated": results["configs_evaluated"],
                        "total_games": results["total_games"],
                        "config_results": results["config_results"],
                    },
                    context="TournamentDaemon",
                )
            except (ImportError, AttributeError):
                # Event type may not exist yet - emit generic
                safe_emit_event(
                    "TOPN_ROUNDROBIN_COMPLETED",
                    {
                        "configs_evaluated": results["configs_evaluated"],
                        "total_games": results["total_games"],
                    },
                    context="TournamentDaemon",
                )

        except ImportError as e:
            logger.warning(f"Top-N round-robin dependencies not available: {e}")
            results["error"] = "import_error"
        except Exception as e:
            logger.error(f"Top-N round-robin tournament failed: {e}")
            results["error"] = str(e)

        return results

    def get_status(self) -> dict[str, Any]:
        """Get current daemon status.

        Returns:
            Status dict with stats and configuration
        """
        # Combine HandlerBase stats with daemon-specific stats
        base_status = {
            "node_id": self.node_id,
            "running": self._running,
            "subscribed": self.is_subscribed,
            "queue_size": self._evaluation_queue.qsize(),
            "cycle_count": self._cycle_count,
            "error_count": len(self._recent_errors),
            "stats": {
                "tournaments_completed": self._tournament_stats.tournaments_completed,
                "games_played": self._tournament_stats.games_played,
                "evaluations_triggered": self._tournament_stats.evaluations_triggered,
                "event_triggers": self._tournament_stats.event_triggers,
                "last_tournament_time": self._tournament_stats.last_tournament_time,
                "last_evaluation_time": self._tournament_stats.last_evaluation_time,
                "recent_errors": self._tournament_stats.errors[-5:],
            },
            "config": {
                "trigger_on_training_completed": self.config.trigger_on_training_completed,
                "enable_periodic_ladder": self.config.enable_periodic_ladder,
                "ladder_interval_seconds": self.config.ladder_interval_seconds,
                "games_per_baseline": self.config.games_per_baseline,
                "baselines": self.config.baselines,
                "enable_topn_roundrobin": self.config.enable_topn_roundrobin,
                "topn_roundrobin_interval_seconds": self.config.topn_roundrobin_interval_seconds,
                "topn_roundrobin_n": self.config.topn_roundrobin_n,
            },
        }
        return base_status

    def health_check(self) -> HealthCheckResult:
        """Check daemon health (January 2026: HandlerBase pattern).

        Returns:
            HealthCheckResult with status and details
        """
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="Tournament daemon not running",
            )

        # Check for high error rate (use HandlerBase error tracking)
        error_count = len(self._recent_errors)
        if error_count > 10:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"Tournament daemon has {error_count} recent errors",
                details=self.get_status(),
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Tournament daemon running ({self._tournament_stats.games_played} games played)",
            details=self.get_status(),
        )


# Module-level singleton functions (delegate to HandlerBase pattern)


def get_tournament_daemon(
    config: TournamentDaemonConfig | None = None,
) -> TournamentDaemon:
    """Get the singleton TournamentDaemon instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        TournamentDaemon instance

    January 2026: Now delegates to HandlerBase.get_instance() singleton pattern.
    """
    return TournamentDaemon.get_instance(config)


def reset_tournament_daemon() -> None:
    """Reset the singleton (for testing).

    January 2026: Now delegates to HandlerBase.reset_instance().
    """
    TournamentDaemon.reset_instance()
