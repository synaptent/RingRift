"""Engine selection and diversity tracking for selfplay scheduling.

Extracted from selfplay_scheduler.py for modularity (P0 decomposition).
Contains engine mode selection logic, engine mix definitions, and
diversity tracking metrics.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)


@dataclass
class DiversityMetrics:
    """Diversity tracking metrics for selfplay scheduling.

    January 2026 Sprint 10: Added opponent_types_by_config tracking for
    diversity maximization. Tracks which opponent types each config has
    played against to ensure variety in training data.
    """

    games_by_engine_mode: dict[str, int] = field(default_factory=dict)
    games_by_board_config: dict[str, int] = field(default_factory=dict)
    games_by_difficulty: dict[str, int] = field(default_factory=dict)
    asymmetric_games: int = 0
    symmetric_games: int = 0
    last_reset: float = field(default_factory=time.time)
    # January 2026 Sprint 10: Track opponent types per config for diversity
    # Key: config_key, Value: set of opponent types (harness names)
    opponent_types_by_config: dict[str, set[str]] = field(default_factory=dict)
    # Track total games per opponent type (cluster-wide)
    games_by_opponent_type: dict[str, int] = field(default_factory=dict)

    def get_diversity_score(self, config_key: str) -> float:
        """Get diversity score for a config (0-1, higher = more diverse).

        January 2026 Sprint 10: Computes diversity based on opponent variety.
        A config that has played against many different opponent types gets
        a higher score.
        """
        # Total possible opponent types (from AI_HARNESS_CONFIGS)
        # Approximate based on typical harness count
        MAX_OPPONENT_TYPES = 8  # random, heuristic, gumbel, minimax, maxn, brs, policy, descent

        opponents_seen = self.opponent_types_by_config.get(config_key, set())
        num_seen = len(opponents_seen)

        if num_seen == 0:
            return 0.0  # No opponents seen = needs diversity
        if num_seen >= MAX_OPPONENT_TYPES:
            return 1.0  # All opponent types seen

        return min(1.0, num_seen / MAX_OPPONENT_TYPES)

    def record_opponent(self, config_key: str, opponent_type: str) -> None:
        """Record that a config played against an opponent type.

        January 2026 Sprint 10: Tracks opponent variety for diversity scoring.
        """
        if config_key not in self.opponent_types_by_config:
            self.opponent_types_by_config[config_key] = set()
        self.opponent_types_by_config[config_key].add(opponent_type)

        # Also track total games by opponent type
        if opponent_type not in self.games_by_opponent_type:
            self.games_by_opponent_type[opponent_type] = 0
        self.games_by_opponent_type[opponent_type] += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with computed metrics."""
        total_games = self.asymmetric_games + self.symmetric_games
        asymmetric_ratio = (
            self.asymmetric_games / total_games if total_games > 0 else 0.0
        )

        engine_total = sum(self.games_by_engine_mode.values())
        engine_mode_distribution = (
            {k: v / engine_total for k, v in self.games_by_engine_mode.items()}
            if engine_total > 0
            else {}
        )

        # January 2026 Sprint 10: Include diversity scores per config
        diversity_scores = {
            config: self.get_diversity_score(config)
            for config in self.opponent_types_by_config.keys()
        }

        return {
            "games_by_engine_mode": dict(self.games_by_engine_mode),
            "games_by_board_config": dict(self.games_by_board_config),
            "games_by_difficulty": dict(self.games_by_difficulty),
            "asymmetric_games": self.asymmetric_games,
            "symmetric_games": self.symmetric_games,
            "asymmetric_ratio": asymmetric_ratio,
            "engine_mode_distribution": engine_mode_distribution,
            "uptime_seconds": time.time() - self.last_reset,
            "games_by_opponent_type": dict(self.games_by_opponent_type),
            "diversity_scores": diversity_scores,
        }


class EngineSelectionMixin:
    """Mixin providing engine selection and diversity tracking for SelfplayScheduler.

    Contains:
    - Engine mix definitions (GPU/CPU variants for large/standard boards)
    - Mode-specific engine mixes (minimax-only, mcts-only, descent-only)
    - Board engine selection logic
    - Engine mode GPU requirement checks
    - Engine mode resolution for category modes
    - Diversity tracking methods

    This mixin does NOT inherit from any base class. Methods using `self`
    will access attributes from the class this is mixed into.
    """

    # GPU-required engine modes (require CUDA or MPS) - December 2025
    # These modes use neural network inference and require GPU acceleration
    GPU_REQUIRED_ENGINE_MODES = {
        "gumbel-mcts", "mcts", "mcts-only", "nnue-guided", "policy-only",
        "nn-minimax", "nn-descent", "gnn", "hybrid",
        "gmo", "ebmo", "ig-gmo", "cage",
    }

    # CPU-compatible engine modes (can run on any node)
    CPU_COMPATIBLE_ENGINE_MODES = {
        "heuristic-only", "heuristic", "random", "random-only",
        "descent-only", "maxn", "brs", "mixed", "diverse",
        "tournament-varied", "heuristic-vs-mcts", "cross-ai",
    }

    # December 2025: Large board engine mix for square19 and hexagonal
    # Uses a weighted mix of engines optimized for large board selfplay:
    # - heuristic: Fast bootstrap, lowest quality but very fast
    # - brs: Best Reply Search, good for 3-4 player
    # - maxn: MaxN search, highest heuristic quality
    # - policy-only: Neural-guided, needs model - GPU required
    # - gumbel-mcts: Balanced neural search with budget 64 - GPU required
    # Dec 31, 2025: Minimum 15% Gumbel MCTS for quality training data (48h autonomous operation)
    # Dec 31, 2025: Added MINIMAX (paranoid) to all engine mixes
    # Paranoid minimax assumes all opponents ally against current player - works for 2-4 players
    LARGE_BOARD_ENGINE_MIX = [
        # (engine_mode, weight, gpu_required, extra_args)
        # Jan 2026: GPU Gumbel MCTS prioritized for high-quality training data
        ("random", 3, False, None),  # 3% - baseline diversity (vs pure random)
        ("heuristic", 10, False, None),  # 10% - fast bootstrap (reduced)
        ("minimax", 5, False, {"depth": 3}),  # 5% - paranoid search (works for 2-4p)
        ("brs", 7, False, None),  # 7% - good for multiplayer
        ("maxn", 5, False, None),  # 5% - highest heuristic quality
        ("nn-descent", 5, True, None),  # 5% - exploration via neural descent (GPU)
        ("policy-only", 15, True, None),  # 15% - neural guided (GPU)
        ("gumbel-mcts", 50, True, {"budget": 128}),  # 50% - neural MCTS (GPU). Mar 29: reduced 800→128 for throughput. At 800 sims with GPU contention, games took 10+ hours and never completed. At 128, games complete in ~30 min. AlphaZero used 64 for throughput mode.
    ]

    # CPU-only variant for nodes without GPU
    LARGE_BOARD_ENGINE_MIX_CPU = [
        # (engine_mode, weight, gpu_required, extra_args)
        # Jan 2026: Full harness diversity for comprehensive training data
        ("random", 5, False, None),  # 5% - baseline diversity (vs pure random)
        ("heuristic", 35, False, None),  # 35% - fast bootstrap
        ("minimax", 15, False, {"depth": 3}),  # 15% - paranoid search (2-4p)
        ("brs", 25, False, None),  # 25% - good for multiplayer
        ("maxn", 20, False, None),  # 20% - highest heuristic quality
    ]

    # December 2025: Standard board engine mix for smaller boards (hex8, square8)
    # Higher neural network weight since games are faster on smaller boards
    # Jan 2026: GPU Gumbel MCTS heavily prioritized for high-quality training data
    STANDARD_BOARD_ENGINE_MIX = [
        # (engine_mode, weight, gpu_required, extra_args)
        # Jan 2026: GPU Gumbel MCTS prioritized for high-quality training data
        ("random", 2, False, None),  # 2% - baseline diversity (vs pure random)
        ("heuristic", 8, False, None),  # 8% - fast bootstrap (reduced)
        ("minimax", 5, False, {"depth": 4}),  # 5% - paranoid search (2-4p, deeper on small boards)
        ("brs", 5, False, None),  # 5% - good for multiplayer diversity
        ("maxn", 5, False, None),  # 5% - highest heuristic quality
        ("nn-descent", 5, True, None),  # 5% - exploration via neural descent (GPU)
        ("policy-only", 10, True, None),  # 10% - neural guided (GPU)
        ("gumbel-mcts", 60, True, {"budget": 128}),  # 60% - HIGH QUALITY neural (GPU) - primary mode
    ]

    # CPU-only variant for standard boards
    STANDARD_BOARD_ENGINE_MIX_CPU = [
        # (engine_mode, weight, gpu_required, extra_args)
        # Jan 2026: Full harness diversity for comprehensive training data
        ("random", 5, False, None),  # 5% - baseline diversity (vs pure random)
        ("heuristic", 25, False, None),  # 25% - fast bootstrap
        ("minimax", 25, False, {"depth": 4}),  # 25% - paranoid search (2-4p)
        ("brs", 25, False, None),  # 25% - good for multiplayer
        ("maxn", 20, False, None),  # 20% - highest heuristic quality
    ]

    # Large board types that should use the large board engine mix
    LARGE_BOARD_TYPES = {"square19", "hexagonal"}
    # Standard board types that should use the standard board engine mix
    STANDARD_BOARD_TYPES = {"hex8", "square8"}

    # Jan 2026: Mode-specific engine mixes for diversity tracking
    # These allow BRS and MaxN to be tracked as distinct modes when using category modes

    # Minimax-only: All heuristic-based search algorithms (minimax, BRS, MaxN)
    MINIMAX_ONLY_ENGINE_MIX = [
        # (engine_mode, weight, gpu_required, extra_args)
        ("minimax", 35, False, {"depth": 4}),  # 35% - paranoid minimax
        ("brs", 35, False, None),  # 35% - Best Reply Search
        ("maxn", 30, False, None),  # 30% - Multi-player MaxN
    ]

    # MCTS-only: All tree search variants (including BRS/MaxN for diversity)
    MCTS_ONLY_ENGINE_MIX = [
        # (engine_mode, weight, gpu_required, extra_args)
        ("gumbel-mcts", 50, True, {"budget": 128}),  # 50% - primary MCTS
        ("minimax", 15, False, {"depth": 3}),  # 15% - tree search variant
        ("brs", 20, False, None),  # 20% - BRS for diversity
        ("maxn", 15, False, None),  # 15% - MaxN for diversity
    ]

    # MCTS-only CPU variant (no GPU)
    MCTS_ONLY_ENGINE_MIX_CPU = [
        # (engine_mode, weight, gpu_required, extra_args)
        ("minimax", 35, False, {"depth": 4}),  # 35% - tree search
        ("brs", 35, False, None),  # 35% - BRS for diversity
        ("maxn", 30, False, None),  # 30% - MaxN for diversity
    ]

    # Descent-only: Neural descent with BRS/MaxN for comparison baseline
    DESCENT_ONLY_ENGINE_MIX = [
        # (engine_mode, weight, gpu_required, extra_args)
        ("nn-descent", 50, True, None),  # 50% - primary neural descent
        ("policy-only", 20, True, None),  # 20% - policy network
        ("brs", 15, False, None),  # 15% - BRS baseline
        ("maxn", 15, False, None),  # 15% - MaxN baseline
    ]

    # Descent-only CPU variant (no GPU)
    DESCENT_ONLY_ENGINE_MIX_CPU = [
        # (engine_mode, weight, gpu_required, extra_args)
        ("heuristic", 40, False, None),  # 40% - heuristic fallback
        ("brs", 30, False, None),  # 30% - BRS baseline
        ("maxn", 30, False, None),  # 30% - MaxN baseline
    ]

    # Mode-specific engine mixes mapping
    MODE_SPECIFIC_MIXES = {
        "minimax-only": (MINIMAX_ONLY_ENGINE_MIX, MINIMAX_ONLY_ENGINE_MIX),  # (GPU, CPU)
        "mcts-only": (MCTS_ONLY_ENGINE_MIX, MCTS_ONLY_ENGINE_MIX_CPU),
        "descent-only": (DESCENT_ONLY_ENGINE_MIX, DESCENT_ONLY_ENGINE_MIX_CPU),
    }

    @classmethod
    def _select_board_engine(
        cls,
        has_gpu: bool,
        board_type: str,
        num_players: int = 0,  # Kept for API compatibility, not used
    ) -> tuple[str, dict[str, Any] | None]:
        """Select an engine mode from the appropriate engine mix for the board type.

        Uses weighted random selection from the engine mix matching the board type:
        - Large boards (square19, hexagonal): LARGE_BOARD_ENGINE_MIX
        - Standard boards (hex8, square8): STANDARD_BOARD_ENGINE_MIX

        All mixes include MINIMAX (paranoid search) which works for 2-4 players.
        Paranoid minimax assumes all opponents ally against the current player.

        GPU vs CPU variants are selected based on node capability.

        Args:
            has_gpu: Whether the node has GPU capability
            board_type: Board type to select engine for
            num_players: Kept for API compatibility (MINIMAX works for all player counts)

        Returns:
            Tuple of (engine_mode, extra_args) where extra_args may contain
            additional parameters like {"budget": 64} for gumbel-mcts.

        December 2025: Extended to support mixed-engine strategy for ALL board types,
        not just large boards. BRS and MaxN now available for hex8/square8 diversity.
        December 31, 2025: Added MINIMAX (paranoid) to all mixes - works for 2-4 players.
        """
        # Select appropriate engine mix based on board type and GPU availability
        if board_type in cls.LARGE_BOARD_TYPES:
            engine_mix = cls.LARGE_BOARD_ENGINE_MIX if has_gpu else cls.LARGE_BOARD_ENGINE_MIX_CPU
            board_category = "large"
        else:
            # Use standard board mix for all other boards (hex8, square8, etc.)
            engine_mix = cls.STANDARD_BOARD_ENGINE_MIX if has_gpu else cls.STANDARD_BOARD_ENGINE_MIX_CPU
            board_category = "standard"

        # Build weighted selection list
        weighted_engines: list[tuple[str, dict[str, Any] | None]] = []
        for engine_mode, weight, gpu_required, extra_args in engine_mix:
            # Skip GPU-required engines on CPU-only nodes
            if gpu_required and not has_gpu:
                continue
            # Add engine with its weight
            weighted_engines.extend([(engine_mode, extra_args)] * weight)

        if not weighted_engines:
            # Fallback to heuristic if no engines available
            logger.warning(
                f"No compatible engines for {board_category} board (gpu={has_gpu}), "
                f"falling back to heuristic"
            )
            return ("heuristic", None)

        # Random weighted selection
        selected = random.choice(weighted_engines)
        engine_mode, extra_args = selected

        logger.debug(
            f"Selected engine '{engine_mode}' for {board_category} board "
            f"(board={board_type}, gpu={has_gpu}, extra_args={extra_args})"
        )

        return (engine_mode, extra_args)

    @classmethod
    def _select_large_board_engine(
        cls,
        has_gpu: bool,
        board_type: str | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        """Legacy wrapper - use _select_board_engine() instead.

        Kept for backward compatibility.
        """
        return cls._select_board_engine(has_gpu, board_type or "square19")

    def _engine_mode_requires_gpu(self, engine_mode: str) -> bool:
        """Check if an engine mode requires GPU acceleration.

        Args:
            engine_mode: The engine mode string (e.g., "gumbel-mcts", "heuristic-only")

        Returns:
            True if the engine mode requires GPU (CUDA or MPS), False otherwise.

        December 2025: Added to ensure GPU-required selfplay is only assigned
        to GPU-capable nodes, preventing wasted compute.
        """
        if not engine_mode:
            return False
        mode_lower = engine_mode.lower().strip()
        return mode_lower in self.GPU_REQUIRED_ENGINE_MODES

    def resolve_engine_mode(
        self,
        engine_mode: str,
        has_gpu: bool = True,
    ) -> tuple[str, dict[str, Any] | None]:
        """Resolve a mode-specific engine to a specific engine from the mix.

        This allows mode categories like 'minimax-only', 'mcts-only', 'descent-only'
        to be resolved to specific engines (brs, maxn, minimax, etc.) for diversity tracking.

        Args:
            engine_mode: The engine mode to resolve (e.g., 'minimax-only', 'mixed')
            has_gpu: Whether the node has GPU capability

        Returns:
            Tuple of (resolved_engine, extra_args) where extra_args may be None

        Example:
            >>> scheduler.resolve_engine_mode('minimax-only', has_gpu=False)
            ('brs', None)  # or ('maxn', None) or ('minimax', {'depth': 4})
        """
        # Check mode-specific mixes first
        if engine_mode in self.MODE_SPECIFIC_MIXES:
            gpu_mix, cpu_mix = self.MODE_SPECIFIC_MIXES[engine_mode]
            engine_mix = gpu_mix if has_gpu else cpu_mix

            # Filter to available engines (respect GPU requirements)
            available_engines = [
                (mode, weight, gpu_required, args)
                for mode, weight, gpu_required, args in engine_mix
                if not gpu_required or has_gpu
            ]

            if available_engines:
                # Weighted random selection
                weighted_engines = []
                for mode, weight, _gpu, args in available_engines:
                    weighted_engines.extend([(mode, args)] * weight)

                if weighted_engines:
                    return random.choice(weighted_engines)

        # For 'mixed' or 'diverse', use board engine selection (defaults to standard mix)
        if engine_mode in ("mixed", "diverse"):
            return self._select_board_engine(has_gpu=has_gpu, board_type="hex8")

        # Not a mode-specific engine, return as-is
        return (engine_mode, None)

    def track_diversity(self, config: dict[str, Any]) -> None:
        """Track diversity metrics for a scheduled selfplay game.

        Args:
            config: Selfplay configuration dict with engine_mode, board_type, num_players, etc.

        January 2026 Sprint 10: Added opponent type tracking for diversity maximization.
        Records which opponent types each config has played against.
        """
        # Track engine mode
        engine_mode = config.get("engine_mode", "unknown")
        if engine_mode not in self.diversity_metrics.games_by_engine_mode:
            self.diversity_metrics.games_by_engine_mode[engine_mode] = 0
        self.diversity_metrics.games_by_engine_mode[engine_mode] += 1

        # Track board config
        board_key = (
            f"{config.get('board_type', 'unknown')}_{config.get('num_players', 0)}p"
        )
        if board_key not in self.diversity_metrics.games_by_board_config:
            self.diversity_metrics.games_by_board_config[board_key] = 0
        self.diversity_metrics.games_by_board_config[board_key] += 1

        # January 2026 Sprint 10: Track opponent type for diversity scoring
        # Record the engine_mode as the opponent type for this config
        self.diversity_metrics.record_opponent(board_key, engine_mode)

        # Track asymmetric vs symmetric
        if config.get("asymmetric"):
            self.diversity_metrics.asymmetric_games += 1
            strong = config.get("strong_config", {})
            weak = config.get("weak_config", {})
            logger.info(
                f"DIVERSE: Asymmetric game scheduled - "
                f"Strong({strong.get('engine_mode')}@D{strong.get('difficulty')}) vs "
                f"Weak({weak.get('engine_mode')}@D{weak.get('difficulty')}) "
                f"on {board_key}"
            )
            # Record both opponent types in asymmetric games
            if strong.get("engine_mode"):
                self.diversity_metrics.record_opponent(board_key, strong["engine_mode"])
            if weak.get("engine_mode"):
                self.diversity_metrics.record_opponent(board_key, weak["engine_mode"])
        else:
            self.diversity_metrics.symmetric_games += 1

        # Track difficulty if available
        difficulty = config.get("difficulty", config.get("difficulty_band"))
        if difficulty:
            diff_key = str(difficulty)
            if diff_key not in self.diversity_metrics.games_by_difficulty:
                self.diversity_metrics.games_by_difficulty[diff_key] = 0
            self.diversity_metrics.games_by_difficulty[diff_key] += 1

    def get_diversity_metrics(self) -> dict[str, Any]:
        """Get diversity tracking metrics for monitoring.

        Returns:
            Dictionary with diversity metrics including computed statistics
        """
        return self.diversity_metrics.to_dict()
