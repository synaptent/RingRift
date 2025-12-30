"""AI Harness abstraction layer for RingRift.

This package provides a unified interface for evaluating AI models under
different search algorithms (harnesses). Each harness can support different
model types (NN, NNUE) and produces standardized evaluation metadata.

Key Components:
- `AIHarness`: Abstract base class for all harness implementations
- `EvaluationMetadata`: Dataclass capturing move selection and search metadata
- `HarnessRegistry`: Factory and compatibility matrix for harness creation

Usage:
    from app.ai.harness import (
        HarnessType,
        create_harness,
        get_harness_compatibility,
        EvaluationMetadata,
    )

    # Create a harness for a specific model
    harness = create_harness(
        harness_type=HarnessType.GUMBEL_MCTS,
        model_path="models/canonical_hex8_2p.pth",
        board_type=BoardType.HEX8,
        num_players=2,
    )

    # Evaluate a position
    move, metadata = harness.evaluate(game_state)

    # Check compatibility
    if get_harness_compatibility(HarnessType.MINIMAX).supports_nnue:
        # Use NNUE with minimax
        ...

Dec 2025: Phase 1 of unified NN/NNUE multi-harness evaluation system.
"""

from .base_harness import AIHarness, HarnessType, ModelType
from .evaluation_metadata import EvaluationMetadata
from .harness_registry import (
    HarnessCompatibility,
    create_harness,
    get_all_harness_types,
    get_compatible_harnesses,
    get_harness_compatibility,
)

__all__ = [
    # Core types
    "AIHarness",
    "HarnessType",
    "ModelType",
    "EvaluationMetadata",
    # Registry functions
    "HarnessCompatibility",
    "create_harness",
    "get_harness_compatibility",
    "get_compatible_harnesses",
    "get_all_harness_types",
]
