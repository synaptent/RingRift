"""Harness registry with factory and compatibility matrix.

This module provides:
- Compatibility matrix defining which model types work with which harnesses
- Factory function for creating harness instances
- Discovery functions for available harnesses
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base_harness import AIHarness, HarnessConfig, HarnessType, ModelType
from .evaluation_metadata import EvaluationMetadata

if TYPE_CHECKING:
    from ...models import BoardType, GameState, Move

logger = logging.getLogger(__name__)


@dataclass
class HarnessCompatibility:
    """Compatibility specification for a harness type.

    Attributes:
        harness_type: The harness type this spec describes.
        supports_nn: Whether full neural networks (v2-v6) are supported.
        supports_nnue: Whether NNUE evaluation is supported.
        requires_policy_head: Whether the harness requires policy output.
        optimal_for: List of use cases this harness excels at.
        min_difficulty: Minimum recommended difficulty level.
        description: Human-readable description of the harness.
    """

    harness_type: HarnessType
    supports_nn: bool
    supports_nnue: bool
    requires_policy_head: bool
    optimal_for: list[str]
    min_difficulty: int = 1
    description: str = ""


# Compatibility matrix defining what each harness supports
HARNESS_COMPATIBILITY: dict[HarnessType, HarnessCompatibility] = {
    HarnessType.GUMBEL_MCTS: HarnessCompatibility(
        harness_type=HarnessType.GUMBEL_MCTS,
        supports_nn=True,
        supports_nnue=False,  # Needs policy head
        requires_policy_head=True,
        optimal_for=["training_data", "high_quality_play"],
        min_difficulty=3,
        description="Gumbel AlphaZero MCTS with Sequential Halving",
    ),
    HarnessType.GPU_GUMBEL: HarnessCompatibility(
        harness_type=HarnessType.GPU_GUMBEL,
        supports_nn=True,
        supports_nnue=False,  # Needs policy head
        requires_policy_head=True,
        optimal_for=["high_throughput_selfplay", "batch_evaluation"],
        min_difficulty=4,
        description="GPU-accelerated Gumbel MCTS for batch selfplay",
    ),
    HarnessType.MINIMAX: HarnessCompatibility(
        harness_type=HarnessType.MINIMAX,
        supports_nn=False,  # Jan 2026: NNUE-only - uses alpha-beta with NNUE position eval
        supports_nnue=True,  # Primary use case
        requires_policy_head=False,
        optimal_for=["2_player_games", "fast_evaluation"],
        min_difficulty=2,
        description="Alpha-beta minimax with NNUE/heuristic evaluation",
    ),
    HarnessType.MAXN: HarnessCompatibility(
        harness_type=HarnessType.MAXN,
        supports_nn=False,  # Jan 2026: NNUE-only - doesn't use NN value head properly
        supports_nnue=True,  # Dec 2025: Added NNUE support
        requires_policy_head=False,
        optimal_for=["multiplayer_games", "3_4_player"],
        min_difficulty=3,
        description="Max-N search for multiplayer (each player maximizes own score)",
    ),
    HarnessType.BRS: HarnessCompatibility(
        harness_type=HarnessType.BRS,
        supports_nn=False,  # Jan 2026: NNUE-only - doesn't use NN value head properly
        supports_nnue=True,  # Dec 2025: Added NNUE support
        requires_policy_head=False,
        optimal_for=["fast_multiplayer", "3_4_player"],
        min_difficulty=2,
        description="Best-Reply Search - fast greedy multiplayer search",
    ),
    HarnessType.POLICY_ONLY: HarnessCompatibility(
        harness_type=HarnessType.POLICY_ONLY,
        supports_nn=True,
        supports_nnue=True,  # NNUE with policy head
        requires_policy_head=True,
        optimal_for=["fast_play", "baseline"],
        min_difficulty=1,
        description="Direct policy sampling without search",
    ),
    HarnessType.DESCENT: HarnessCompatibility(
        harness_type=HarnessType.DESCENT,
        supports_nn=True,
        supports_nnue=False,  # Needs full NN for gradient descent
        requires_policy_head=True,
        optimal_for=["exploration", "research"],
        min_difficulty=4,
        description="Gradient descent on value function for move selection",
    ),
    HarnessType.HEURISTIC: HarnessCompatibility(
        harness_type=HarnessType.HEURISTIC,
        supports_nn=False,  # No NN needed
        supports_nnue=False,  # No NN needed
        requires_policy_head=False,
        optimal_for=["baseline", "fast_bootstrap"],
        min_difficulty=1,
        description="Hand-crafted heuristic evaluation only",
    ),
    HarnessType.RANDOM: HarnessCompatibility(
        harness_type=HarnessType.RANDOM,
        supports_nn=False,  # No model needed
        supports_nnue=False,  # No model needed
        requires_policy_head=False,
        optimal_for=["baseline", "sanity_check", "diversity"],
        min_difficulty=1,
        description="Uniform random move selection for baseline comparison",
    ),
}


def get_harness_compatibility(harness_type: HarnessType) -> HarnessCompatibility:
    """Get compatibility specification for a harness type.

    Args:
        harness_type: The harness type to query.

    Returns:
        HarnessCompatibility spec for the harness.

    Raises:
        KeyError: If harness type is not in registry.
    """
    return HARNESS_COMPATIBILITY[harness_type]


def get_compatible_harnesses(
    model_type: ModelType,
    requires_policy: bool | None = None,
) -> list[HarnessType]:
    """Get all harnesses compatible with a model type.

    Args:
        model_type: The model type to check compatibility for.
        requires_policy: If True, only return harnesses that use policy.
                        If False, only return those that don't require it.
                        If None, return all compatible harnesses.

    Returns:
        List of compatible HarnessType values.
    """
    compatible = []
    for harness_type, compat in HARNESS_COMPATIBILITY.items():
        # Check model type compatibility
        if model_type == ModelType.NEURAL_NET and not compat.supports_nn:
            continue
        if model_type == ModelType.NNUE and not compat.supports_nnue:
            continue
        if model_type == ModelType.HEURISTIC:
            # Heuristic works with all harnesses (as fallback)
            pass

        # Check policy requirement filter
        if requires_policy is not None:
            if requires_policy and not compat.requires_policy_head:
                continue
            if not requires_policy and compat.requires_policy_head:
                continue

        compatible.append(harness_type)

    return compatible


def get_all_harness_types() -> list[HarnessType]:
    """Get all available harness types."""
    return list(HARNESS_COMPATIBILITY.keys())


# Harness implementation classes (lazy imported to avoid circular deps)
_HARNESS_IMPLEMENTATIONS: dict[HarnessType, type[AIHarness]] = {}


def _ensure_implementations_loaded() -> None:
    """Lazily load harness implementations."""
    global _HARNESS_IMPLEMENTATIONS
    if _HARNESS_IMPLEMENTATIONS:
        return

    # Import implementations
    from .implementations import (
        BRSHarness,
        DescentHarness,
        GumbelMCTSHarness,
        GPUGumbelHarness,
        HeuristicHarness,
        MaxNHarness,
        MinimaxHarness,
        PolicyOnlyHarness,
        RandomHarness,
    )

    _HARNESS_IMPLEMENTATIONS = {
        HarnessType.GUMBEL_MCTS: GumbelMCTSHarness,
        HarnessType.GPU_GUMBEL: GPUGumbelHarness,
        HarnessType.MINIMAX: MinimaxHarness,
        HarnessType.MAXN: MaxNHarness,
        HarnessType.BRS: BRSHarness,
        HarnessType.POLICY_ONLY: PolicyOnlyHarness,
        HarnessType.DESCENT: DescentHarness,
        HarnessType.HEURISTIC: HeuristicHarness,
        HarnessType.RANDOM: RandomHarness,
    }


def create_harness(
    harness_type: HarnessType,
    model_path: str | Path | None = None,
    model_id: str = "",
    board_type: BoardType | None = None,
    num_players: int = 2,
    difficulty: int = 5,
    think_time_ms: int | None = None,
    simulations: int = 200,
    depth: int = 3,
    model_type: ModelType | None = None,
    extra: dict[str, Any] | None = None,
) -> AIHarness:
    """Create a harness instance.

    This is the main factory function for creating harnesses. It validates
    compatibility and creates the appropriate implementation.

    Args:
        harness_type: Type of harness to create.
        model_path: Path to model checkpoint (optional for heuristic).
        model_id: Identifier for Elo tracking.
        board_type: Board type for model compatibility.
        num_players: Number of players (2, 3, or 4).
        difficulty: AI difficulty level (1-10).
        think_time_ms: Maximum time per move in milliseconds.
        simulations: Number of MCTS simulations (for tree search).
        depth: Search depth (for minimax-style search).
        model_type: Explicit model type (auto-detected from path if None).
        extra: Additional harness-specific options.

    Returns:
        Configured AIHarness instance.

    Raises:
        ValueError: If model type is incompatible with harness.
    """
    # Determine model type
    if model_type is None:
        if model_path is None:
            model_type = ModelType.HEURISTIC
        elif "nnue" in str(model_path).lower():
            model_type = ModelType.NNUE
        else:
            model_type = ModelType.NEURAL_NET

    # Validate compatibility
    compat = get_harness_compatibility(harness_type)
    if model_type == ModelType.NEURAL_NET and not compat.supports_nn:
        raise ValueError(
            f"Harness {harness_type.value} does not support full neural networks"
        )
    if model_type == ModelType.NNUE and not compat.supports_nnue:
        raise ValueError(
            f"Harness {harness_type.value} does not support NNUE"
        )

    # Create config
    config = HarnessConfig(
        harness_type=harness_type,
        model_type=model_type,
        model_path=str(model_path) if model_path else None,
        model_id=model_id or _generate_model_id(model_path, board_type, num_players),
        board_type=board_type,
        num_players=num_players,
        difficulty=difficulty,
        think_time_ms=think_time_ms,
        simulations=simulations,
        depth=depth,
        extra=extra or {},
    )

    # Create harness
    _ensure_implementations_loaded()
    harness_class = _HARNESS_IMPLEMENTATIONS.get(harness_type)
    if harness_class is None:
        raise ValueError(f"No implementation for harness type: {harness_type}")

    return harness_class(config)


def _generate_model_id(
    model_path: str | Path | None,
    board_type: BoardType | None,
    num_players: int,
) -> str:
    """Generate a model ID from path and config."""
    if model_path is None:
        if board_type:
            return f"heuristic_{board_type.value}_{num_players}p"
        return f"heuristic_{num_players}p"

    # Extract from path
    path = Path(model_path)
    stem = path.stem

    # Try to parse canonical naming
    if "canonical_" in stem:
        return stem.replace("canonical_", "ringrift_")

    # Fall back to filename
    return stem


# =============================================================================
# Player Restrictions (Dec 2025 consolidation)
# =============================================================================

# Canonical player count restrictions for harnesses
# Format: (min_players, max_players)
# Note: HarnessType.MCTS does not exist - use GUMBEL_MCTS or GPU_GUMBEL instead
HARNESS_PLAYER_RESTRICTIONS: dict[HarnessType, tuple[int, int]] = {
    HarnessType.GUMBEL_MCTS: (2, 4),
    HarnessType.GPU_GUMBEL: (2, 4),
    HarnessType.POLICY_ONLY: (2, 4),
    HarnessType.DESCENT: (2, 4),
    HarnessType.HEURISTIC: (2, 4),
    HarnessType.MINIMAX: (2, 2),  # Only 2-player (alpha-beta)
    HarnessType.MAXN: (3, 4),  # 3-4 player only (multiplayer search)
    HarnessType.BRS: (3, 4),  # 3-4 player only (best-reply search)
    HarnessType.RANDOM: (2, 4),  # Jan 1, 2026: Random for baseline and diversity
}


def get_harness_player_range(harness_type: HarnessType) -> tuple[int, int]:
    """Get valid player count range for a harness.

    Args:
        harness_type: The harness type to query.

    Returns:
        Tuple of (min_players, max_players).
    """
    return HARNESS_PLAYER_RESTRICTIONS.get(harness_type, (2, 4))


def is_harness_valid_for_player_count(
    harness_type: HarnessType,
    num_players: int,
) -> bool:
    """Check if a harness supports a given player count.

    Args:
        harness_type: The harness type to check.
        num_players: Number of players (2, 3, or 4).

    Returns:
        True if the harness supports this player count.
    """
    min_p, max_p = get_harness_player_range(harness_type)
    return min_p <= num_players <= max_p


def get_harnesses_for_model_and_players(
    model_type: ModelType,
    num_players: int,
) -> list[HarnessType]:
    """Get harnesses compatible with both model type and player count.

    This is the preferred function for gauntlet evaluation as it filters
    by both model compatibility AND player count restrictions.

    Args:
        model_type: The model type (NEURAL_NET, NNUE, HEURISTIC).
        num_players: Number of players (2, 3, or 4).

    Returns:
        List of compatible HarnessType values.
    """
    # First get model-compatible harnesses
    compatible = get_compatible_harnesses(model_type)

    # Then filter by player count
    return [
        h for h in compatible
        if is_harness_valid_for_player_count(h, num_players)
    ]


def get_harness_matrix() -> dict[ModelType, list[HarnessType]]:
    """Get derived harness compatibility matrix by model type.

    This function derives the matrix from the canonical HARNESS_COMPATIBILITY
    dataclass entries rather than maintaining a duplicate definition.

    Returns:
        Dictionary mapping ModelType to list of compatible HarnessType.
    """
    matrix: dict[ModelType, list[HarnessType]] = {
        ModelType.NEURAL_NET: [],
        ModelType.NNUE: [],
        ModelType.HEURISTIC: [],
    }

    for harness_type, compat in HARNESS_COMPATIBILITY.items():
        if compat.supports_nn:
            matrix[ModelType.NEURAL_NET].append(harness_type)
        if compat.supports_nnue:
            matrix[ModelType.NNUE].append(harness_type)
        if harness_type == HarnessType.HEURISTIC:
            matrix[ModelType.HEURISTIC].append(harness_type)

    return matrix


def get_harness_compatibility_dict() -> dict[str, dict[str, Any]]:
    """Get harness compatibility as simple dict for external consumers.

    This is for backwards compatibility with code that expects string keys
    rather than enum types.

    Returns:
        Dictionary mapping harness name to compatibility info.
    """
    result: dict[str, dict[str, Any]] = {}
    for harness_type, compat in HARNESS_COMPATIBILITY.items():
        min_p, max_p = get_harness_player_range(harness_type)
        result[harness_type.value] = {
            "nn": compat.supports_nn,
            "nnue": compat.supports_nnue,
            "policy_required": compat.requires_policy_head,
            "min_players": min_p,
            "max_players": max_p,
            "optimal_for": compat.optimal_for,
            "description": compat.description,
        }
    return result
