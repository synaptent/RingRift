"""Supported neural-network facade for RingRift AI.

This package is the stable public surface for RingRift neural-network models,
encoders, and helper utilities. Active code should import from
``app.ai.neural_net`` (or narrower submodules such as ``app.ai.nnue_policy``
when appropriate), while direct imports from ``app.ai._neural_net_legacy``
are deprecated compatibility paths to the archived monolithic implementation.

The package is organized as follows:
- constants.py: Policy sizes and encoding constants
- blocks.py: Reusable neural network building blocks
- _neural_net_legacy.py: Archived monolithic implementation, still re-exported
  for compatibility while migration drains remaining callers

Public symbols are re-exported here so callers can stay on the supported
package facade while the remaining legacy pieces are retired.

Import layout (PEP 562 lazy exports):
    The torch-free encoding surface (constants, hex/square action encoders,
    canonical move encoding) is imported eagerly. Torch-dependent symbols
    (architectures, blocks, losses, factories, legacy NeuralNetAI) are
    resolved lazily on first attribute access via module ``__getattr__``.
    This keeps rules-only consumers (e.g. the ringrift-env RL environment
    package) importable without torch installed, and follows the repo's
    existing lazy-import-to-avoid-heavy-startup convention.
"""

from importlib import import_module
from typing import Any

# Constants - fully migrated to constants.py (Phase 1 complete). Torch-free.
from app.ai.neural_net.constants import (
    BOARD_POLICY_SIZES,
    BOARD_SPATIAL_SIZES,
    HEX8_BOARD_SIZE,
    HEX_BOARD_SIZE,
    HEX_DIRS,
    HEX_MAX_DIST,
    HEX_MOVEMENT_BASE,
    HEX_MOVEMENT_SPAN,
    HEX_PLACEMENT_SPAN,
    HEX_SPECIAL_BASE,
    INVALID_MOVE_INDEX,
    MAX_DIST_SQUARE8,
    MAX_DIST_SQUARE19,
    MAX_N,
    MAX_PLAYERS,
    NUM_HEX_DIRS,
    NUM_LINE_DIRS,
    NUM_SQUARE_DIRS,
    P_HEX,
    POLICY_SIZE,
    POLICY_SIZE_8x8,
    POLICY_SIZE_19x19,
    POLICY_SIZE_HEX8,
    SQUARE8_FORCED_ELIMINATION_IDX,
    SQUARE8_LINE_CHOICE_BASE,
    SQUARE8_LINE_FORM_BASE,
    SQUARE8_MOVEMENT_BASE,
    SQUARE8_NO_LINE_ACTION_IDX,
    SQUARE8_NO_MOVEMENT_ACTION_IDX,
    SQUARE8_NO_PLACEMENT_ACTION_IDX,
    SQUARE8_NO_TERRITORY_ACTION_IDX,
    SQUARE8_PLACEMENT_SPAN,
    SQUARE8_SKIP_CAPTURE_IDX,
    SQUARE8_SKIP_PLACEMENT_IDX,
    SQUARE8_SKIP_RECOVERY_IDX,
    SQUARE8_SKIP_TERRITORY_PROCESSING_IDX,
    SQUARE8_SWAP_SIDES_IDX,
    SQUARE8_TERRITORY_CHOICE_BASE,
    SQUARE8_TERRITORY_CLAIM_BASE,
    SQUARE19_FORCED_ELIMINATION_IDX,
    SQUARE19_LINE_CHOICE_BASE,
    SQUARE19_LINE_FORM_BASE,
    SQUARE19_NO_LINE_ACTION_IDX,
    SQUARE19_NO_MOVEMENT_ACTION_IDX,
    SQUARE19_NO_PLACEMENT_ACTION_IDX,
    SQUARE19_NO_TERRITORY_ACTION_IDX,
    SQUARE19_SKIP_CAPTURE_IDX,
    SQUARE19_SKIP_PLACEMENT_IDX,
    SQUARE19_SKIP_RECOVERY_IDX,
    SQUARE19_SKIP_TERRITORY_PROCESSING_IDX,
    SQUARE19_SWAP_SIDES_IDX,
    SQUARE19_TERRITORY_CHOICE_BASE,
    SQUARE19_TERRITORY_CLAIM_BASE,
    TERRITORY_MAX_PLAYERS,
    TERRITORY_SIZE_BUCKETS,
    get_policy_size_for_board,
    get_spatial_size_for_board,
)

# Hex encoding - migrated to hex_encoding.py (Phase 2). Torch-free.
from app.ai.neural_net.hex_encoding import (
    ActionEncoderHex,
    _from_canonical_xy,
    _infer_board_size,
    _pos_from_key,
    _to_canonical_xy,
)

# Square encoding - migrated to square_encoding.py (Phase 2). Torch-free.
from app.ai.neural_net.square_encoding import (
    ActionEncoderSquare,
    ActionEncoderSquare8,
    ActionEncoderSquare19,
    DecodedPolicyIndex,
    SQUARE_DIRS,
    get_action_encoder,
    # Legacy wrapper functions (backwards compatible)
    _decode_move_square8,
    _decode_move_square19,
    _encode_move_square8,
    _encode_move_square19,
    transform_policy_index_square,
)

# Lazily-resolved symbols (PEP 562). Maps exported name -> providing module.
# Most entries are torch-dependent; canonical_move_encoding is torch-free but
# must be lazy because it imports app.ai.neural_net.constants, which would
# recurse into this package while it is still initializing.
_LAZY_EXPORTS: dict[str, str] = {
    # Canonical move encoding (migrated from _neural_net_legacy.py)
    "decode_move_for_board": "app.ai.canonical_move_encoding",
    "encode_move_for_board": "app.ai.canonical_move_encoding",
    "get_encoder_for_board": "app.ai.canonical_move_encoding",
    # Building blocks - blocks.py (Phase 1)
    "AttentionResidualBlock": "app.ai.neural_net.blocks",
    "ResidualBlock": "app.ai.neural_net.blocks",
    "SEResidualBlock": "app.ai.neural_net.blocks",
    # Square architectures - square_architectures.py (Phase 2)
    # NOTE: RingRiftCNN_v5 (v5.0.0) was removed Dec 2025 - use RingRiftCNN_v5_Heavy
    "RingRiftCNN_v2": "app.ai.neural_net.square_architectures",
    "RingRiftCNN_v2_Lite": "app.ai.neural_net.square_architectures",
    "RingRiftCNN_v3": "app.ai.neural_net.square_architectures",
    "RingRiftCNN_v3_Flat": "app.ai.neural_net.square_architectures",
    "RingRiftCNN_v3_Lite": "app.ai.neural_net.square_architectures",
    "RingRiftCNN_v4": "app.ai.neural_net.square_architectures",
    # Hex architectures - hex_architectures.py (Phase 2)
    "HexNeuralNet_v2": "app.ai.neural_net.hex_architectures",
    "HexNeuralNet_v2_Lite": "app.ai.neural_net.hex_architectures",
    "HexNeuralNet_v3": "app.ai.neural_net.hex_architectures",
    "HexNeuralNet_v3_Flat": "app.ai.neural_net.hex_architectures",
    "HexNeuralNet_v3_Lite": "app.ai.neural_net.hex_architectures",
    "HexNeuralNet_v4": "app.ai.neural_net.hex_architectures",
    # V5 Heavy architectures (December 2025)
    "HeuristicEncoder": "app.ai.neural_net.v5_heavy",
    "HexNeuralNet_v5_Heavy": "app.ai.neural_net.v5_heavy",
    "RingRiftCNN_v5_Heavy": "app.ai.neural_net.v5_heavy",
    "create_v5_heavy_model": "app.ai.neural_net.v5_heavy",
    "NUM_HEURISTIC_FEATURES": "app.ai.neural_net.v5_heavy",
    # Graph encoding - enables GNN-based position evaluation (December 2025)
    "NODE_FEATURE_IDX": "app.ai.neural_net.graph_encoding",
    "EDGE_ATTR_IDX": "app.ai.neural_net.graph_encoding",
    "board_to_graph": "app.ai.neural_net.graph_encoding",
    "board_to_graph_hex": "app.ai.neural_net.graph_encoding",
    # Loss functions - neural_losses.py (Phase 3)
    "multi_player_value_loss": "app.ai.neural_losses",
    "rank_distribution_loss": "app.ai.neural_losses",
    "ranks_from_game_result": "app.ai.neural_losses",
    # Model factory - model_factory.py (Phase 4)
    "create_model_for_board": "app.ai.neural_net.model_factory",
    "get_memory_tier": "app.ai.neural_net.model_factory",
    "get_model_config_for_board": "app.ai.neural_net.model_factory",
    # Unified neural net factory for tournament daemon
    "UnifiedNeuralNetFactory": "app.ai.unified_factory",
    # Classes and functions still in the archived legacy module. Reachable
    # through app/ai/_neural_net_legacy.py only as a compatibility path.
    "_MODEL_CACHE": "app.ai._neural_net_legacy",
    "NeuralNetAI": "app.ai._neural_net_legacy",
    "clear_model_cache": "app.ai._neural_net_legacy",
    "create_hex_mask": "app.ai._neural_net_legacy",
    "get_cached_model_count": "app.ai._neural_net_legacy",
}


def __getattr__(name: str) -> Any:
    """Resolve torch-dependent exports lazily (PEP 562).

    Keeps `from app.ai.neural_net import <symbol>` working for every symbol
    in ``__all__`` while deferring torch imports until a torch-dependent
    symbol is actually requested.
    """
    target = _LAZY_EXPORTS.get(name)
    if target is not None:
        if target == "app.ai._neural_net_legacy":
            # Suppress the deprecation warning for backwards-compatible
            # re-exports, matching the previous eager-import behaviour.
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                module = import_module(target)
        else:
            module = import_module(target)
        value = getattr(module, name)
        globals()[name] = value  # Cache so subsequent access skips __getattr__
        return value
    # Fall back to submodule access (e.g. app.ai.neural_net.blocks).
    try:
        return import_module(f"{__name__}.{name}")
    except ModuleNotFoundError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


__all__ = [
    "BOARD_POLICY_SIZES",
    "BOARD_SPATIAL_SIZES",
    "HEX8_BOARD_SIZE",
    "HEX_BOARD_SIZE",
    "HEX_DIRS",
    "HEX_MAX_DIST",
    "HEX_MOVEMENT_BASE",
    "HEX_MOVEMENT_SPAN",
    "HEX_PLACEMENT_SPAN",
    "HEX_SPECIAL_BASE",
    "INVALID_MOVE_INDEX",
    "MAX_N",
    "MAX_PLAYERS",
    "NUM_HEX_DIRS",
    "NUM_LINE_DIRS",
    "NUM_SQUARE_DIRS",
    "POLICY_SIZE",
    "POLICY_SIZE_HEX8",
    "P_HEX",
    "SQUARE8_FORCED_ELIMINATION_IDX",
    "SQUARE8_LINE_FORM_BASE",
    "SQUARE8_MOVEMENT_BASE",
    "SQUARE8_NO_LINE_ACTION_IDX",
    "SQUARE8_NO_MOVEMENT_ACTION_IDX",
    "SQUARE8_NO_PLACEMENT_ACTION_IDX",
    "SQUARE8_NO_TERRITORY_ACTION_IDX",
    "SQUARE8_PLACEMENT_SPAN",
    "SQUARE8_SKIP_CAPTURE_IDX",
    "SQUARE8_SKIP_PLACEMENT_IDX",
    "SQUARE8_SKIP_RECOVERY_IDX",
    "SQUARE8_SKIP_TERRITORY_PROCESSING_IDX",
    "SQUARE8_SWAP_SIDES_IDX",
    "SQUARE8_TERRITORY_CLAIM_BASE",
    "SQUARE19_FORCED_ELIMINATION_IDX",
    "SQUARE19_LINE_FORM_BASE",
    "SQUARE19_NO_LINE_ACTION_IDX",
    "SQUARE19_NO_MOVEMENT_ACTION_IDX",
    "SQUARE19_NO_PLACEMENT_ACTION_IDX",
    "SQUARE19_NO_TERRITORY_ACTION_IDX",
    "SQUARE19_SKIP_CAPTURE_IDX",
    "SQUARE19_SKIP_TERRITORY_PROCESSING_IDX",
    "TERRITORY_MAX_PLAYERS",
    "TERRITORY_SIZE_BUCKETS",
    "ActionEncoderHex",
    "ActionEncoderSquare",
    "ActionEncoderSquare8",
    "ActionEncoderSquare19",
    "AttentionResidualBlock",
    "DecodedPolicyIndex",
    "HexNeuralNet_v2",
    "HexNeuralNet_v2_Lite",
    "HexNeuralNet_v3",
    "HexNeuralNet_v3_Flat",
    "HexNeuralNet_v3_Lite",
    "HexNeuralNet_v4",
    "HexNeuralNet_v5_Heavy",
    "HeuristicEncoder",
    "NeuralNetAI",
    "NUM_HEURISTIC_FEATURES",
    "POLICY_SIZE_8x8",
    "POLICY_SIZE_19x19",
    "ResidualBlock",
    "RingRiftCNN_v2",
    "RingRiftCNN_v2_Lite",
    "RingRiftCNN_v3",
    "RingRiftCNN_v3_Flat",
    "RingRiftCNN_v3_Lite",
    "RingRiftCNN_v4",
    "RingRiftCNN_v5_Heavy",
    "SEResidualBlock",
    "create_v5_heavy_model",
    "_decode_move_square8",
    "_decode_move_square19",
    "_encode_move_square8",
    "_encode_move_square19",
    "_from_canonical_xy",
    "_infer_board_size",
    "_pos_from_key",
    "_to_canonical_xy",
    "clear_model_cache",
    "create_hex_mask",
    "create_model_for_board",
    "decode_move_for_board",
    "encode_move_for_board",
    "get_action_encoder",
    "get_cached_model_count",
    "get_encoder_for_board",
    "get_memory_tier",
    "get_model_config_for_board",
    "get_policy_size_for_board",
    "get_spatial_size_for_board",
    "SQUARE_DIRS",
    "multi_player_value_loss",
    "rank_distribution_loss",
    "ranks_from_game_result",
    "transform_policy_index_square",
    "UnifiedNeuralNetFactory",
]


def __dir__() -> list[str]:
    """Expose the intended neural-net package surface for discoverability."""

    return sorted(set(globals()) | set(__all__))
