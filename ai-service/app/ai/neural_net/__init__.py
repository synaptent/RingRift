"""Neural network package for RingRift AI.

This package contains the neural network implementations for the RingRift
AI system, including CNN architectures for different board types and
the NeuralNetAI wrapper class.

The package is organized as follows:
- constants.py: Policy sizes and encoding constants
- blocks.py: Reusable neural network building blocks
- _neural_net_legacy.py: Original monolithic module (being migrated)

For backwards compatibility, all public symbols are re-exported.
Migration status: blocks.py contains extracted building blocks.
"""

# Building blocks - migrated to blocks.py (Phase 1)
from app.ai.neural_net.blocks import (
    AttentionResidualBlock,
    ResidualBlock,
    SEResidualBlock,
)

# Re-export from legacy module for backwards compatibility
# This ensures `from app.ai.neural_net import X` continues to work
from app.ai._neural_net_legacy import (
    BOARD_POLICY_SIZES,
    BOARD_SPATIAL_SIZES,
    # Hex8 constants
    HEX8_BOARD_SIZE,
    HEX_BOARD_SIZE,
    HEX_DIRS,
    HEX_MAX_DIST,
    HEX_MOVEMENT_BASE,
    HEX_MOVEMENT_SPAN,
    HEX_PLACEMENT_SPAN,
    HEX_SPECIAL_BASE,
    # Constants
    INVALID_MOVE_INDEX,
    MAX_N,
    MAX_PLAYERS,
    NUM_HEX_DIRS,
    NUM_LINE_DIRS,
    NUM_SQUARE_DIRS,
    P_HEX,
    POLICY_SIZE,
    POLICY_SIZE_HEX8,
    SQUARE8_FORCED_ELIMINATION_IDX,
    SQUARE8_LINE_FORM_BASE,
    SQUARE8_MOVEMENT_BASE,
    SQUARE8_NO_LINE_ACTION_IDX,
    SQUARE8_NO_MOVEMENT_ACTION_IDX,
    SQUARE8_NO_PLACEMENT_ACTION_IDX,
    SQUARE8_NO_TERRITORY_ACTION_IDX,
    # Square8 layout constants
    SQUARE8_PLACEMENT_SPAN,
    SQUARE8_SKIP_CAPTURE_IDX,
    SQUARE8_SKIP_PLACEMENT_IDX,
    SQUARE8_SKIP_TERRITORY_PROCESSING_IDX,
    SQUARE8_SWAP_SIDES_IDX,
    SQUARE8_TERRITORY_CLAIM_BASE,
    SQUARE19_FORCED_ELIMINATION_IDX,
    # Square19 layout constants
    SQUARE19_LINE_FORM_BASE,
    SQUARE19_NO_LINE_ACTION_IDX,
    SQUARE19_NO_MOVEMENT_ACTION_IDX,
    SQUARE19_NO_PLACEMENT_ACTION_IDX,
    SQUARE19_NO_TERRITORY_ACTION_IDX,
    SQUARE19_SKIP_CAPTURE_IDX,
    SQUARE19_SKIP_TERRITORY_PROCESSING_IDX,
    TERRITORY_MAX_PLAYERS,
    TERRITORY_SIZE_BUCKETS,
    # Hex architectures
    ActionEncoderHex,
    # Dataclasses
    DecodedPolicyIndex,
    HexNeuralNet_v2,
    HexNeuralNet_v2_Lite,
    HexNeuralNet_v3,
    HexNeuralNet_v3_Lite,
    # Main AI class
    NeuralNetAI,
    POLICY_SIZE_8x8,
    POLICY_SIZE_19x19,
    # CNN architectures
    RingRiftCNN_v2,
    RingRiftCNN_v2_Lite,
    RingRiftCNN_v3,
    RingRiftCNN_v3_Lite,
    RingRiftCNN_v4,
    _decode_move_square8,
    _encode_move_square8,
    _from_canonical_xy,
    _pos_from_key,
    # Internal helpers (used by encoding module and tests)
    _to_canonical_xy,
    # Cache functions
    clear_model_cache,
    create_hex_mask,
    create_model_for_board,
    decode_move_for_board,
    encode_move_for_board,
    get_cached_model_count,
    get_memory_tier,
    get_model_config_for_board,
    # Functions
    get_policy_size_for_board,
    get_spatial_size_for_board,
    # Loss functions (re-exported)
    multi_player_value_loss,
    rank_distribution_loss,
    ranks_from_game_result,
    transform_policy_index_square,
)

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
    "AttentionResidualBlock",
    "DecodedPolicyIndex",
    "HexNeuralNet_v2",
    "HexNeuralNet_v2_Lite",
    "HexNeuralNet_v3",
    "HexNeuralNet_v3_Lite",
    "NeuralNetAI",
    "POLICY_SIZE_8x8",
    "POLICY_SIZE_19x19",
    "ResidualBlock",
    "RingRiftCNN_v2",
    "RingRiftCNN_v2_Lite",
    "RingRiftCNN_v3",
    "RingRiftCNN_v3_Lite",
    "RingRiftCNN_v4",
    "SEResidualBlock",
    "_decode_move_square8",
    "_encode_move_square8",
    "_from_canonical_xy",
    "_pos_from_key",
    "_to_canonical_xy",
    "clear_model_cache",
    "create_hex_mask",
    "create_model_for_board",
    "decode_move_for_board",
    "encode_move_for_board",
    "get_cached_model_count",
    "get_memory_tier",
    "get_model_config_for_board",
    "get_policy_size_for_board",
    "get_spatial_size_for_board",
    "multi_player_value_loss",
    "rank_distribution_loss",
    "ranks_from_game_result",
    "transform_policy_index_square",
]
