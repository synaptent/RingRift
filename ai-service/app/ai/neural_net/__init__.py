"""Neural network package for RingRift AI.

This package contains the neural network implementations for the RingRift
AI system, including CNN architectures for different board types and
the NeuralNetAI wrapper class.

The package is organized as follows:
- constants.py: Policy sizes and encoding constants
- blocks.py: Reusable neural network building blocks
- _neural_net_legacy.py: Original monolithic module (being migrated)

For backwards compatibility, all public symbols are re-exported from
the legacy module.
"""

# Import torch for patching compatibility in tests
import torch

# Re-export everything from the legacy module for backwards compatibility
# This ensures `from app.ai.neural_net import X` continues to work
from app.ai._neural_net_legacy import (
    # Constants
    INVALID_MOVE_INDEX,
    MAX_N,
    MAX_PLAYERS,
    POLICY_SIZE,
    POLICY_SIZE_8x8,
    POLICY_SIZE_19x19,
    POLICY_SIZE_HEX8,
    P_HEX,
    HEX_BOARD_SIZE,
    HEX_MAX_DIST,
    HEX_DIRS,
    NUM_HEX_DIRS,
    HEX_PLACEMENT_SPAN,
    HEX_MOVEMENT_BASE,
    HEX_MOVEMENT_SPAN,
    HEX_SPECIAL_BASE,
    NUM_SQUARE_DIRS,
    NUM_LINE_DIRS,
    TERRITORY_SIZE_BUCKETS,
    TERRITORY_MAX_PLAYERS,
    BOARD_POLICY_SIZES,
    BOARD_SPATIAL_SIZES,
    # Square8 layout constants
    SQUARE8_PLACEMENT_SPAN,
    SQUARE8_MOVEMENT_BASE,
    SQUARE8_LINE_FORM_BASE,
    SQUARE8_TERRITORY_CLAIM_BASE,
    SQUARE8_SKIP_PLACEMENT_IDX,
    SQUARE8_SWAP_SIDES_IDX,
    SQUARE8_NO_PLACEMENT_ACTION_IDX,
    SQUARE8_NO_MOVEMENT_ACTION_IDX,
    SQUARE8_SKIP_CAPTURE_IDX,
    SQUARE8_NO_LINE_ACTION_IDX,
    SQUARE8_NO_TERRITORY_ACTION_IDX,
    SQUARE8_SKIP_TERRITORY_PROCESSING_IDX,
    SQUARE8_FORCED_ELIMINATION_IDX,
    # Square19 layout constants
    SQUARE19_LINE_FORM_BASE,
    SQUARE19_NO_PLACEMENT_ACTION_IDX,
    SQUARE19_NO_MOVEMENT_ACTION_IDX,
    SQUARE19_SKIP_CAPTURE_IDX,
    SQUARE19_NO_LINE_ACTION_IDX,
    SQUARE19_NO_TERRITORY_ACTION_IDX,
    SQUARE19_SKIP_TERRITORY_PROCESSING_IDX,
    SQUARE19_FORCED_ELIMINATION_IDX,
    # Hex8 constants
    HEX8_BOARD_SIZE,
    # Dataclasses
    DecodedPolicyIndex,
    # Functions
    get_policy_size_for_board,
    get_spatial_size_for_board,
    encode_move_for_board,
    decode_move_for_board,
    transform_policy_index_square,
    create_hex_mask,
    get_memory_tier,
    create_model_for_board,
    get_model_config_for_board,
    # Building blocks
    ResidualBlock,
    SEResidualBlock,
    AttentionResidualBlock,
    # CNN architectures
    RingRiftCNN_v2,
    RingRiftCNN_v2_Lite,
    RingRiftCNN_v3,
    RingRiftCNN_v3_Lite,
    RingRiftCNN_v4,
    # Hex architectures
    ActionEncoderHex,
    HexNeuralNet_v2,
    HexNeuralNet_v2_Lite,
    HexNeuralNet_v3,
    HexNeuralNet_v3_Lite,
    # Main AI class
    NeuralNetAI,
    # Loss functions (re-exported)
    multi_player_value_loss,
    rank_distribution_loss,
    ranks_from_game_result,
    # Cache functions
    clear_model_cache,
    get_cached_model_count,
    # Internal helpers (used by encoding module and tests)
    _to_canonical_xy,
    _from_canonical_xy,
    _pos_from_key,
    _encode_move_square8,
    _decode_move_square8,
)

__all__ = [
    # Constants
    "INVALID_MOVE_INDEX",
    "MAX_N",
    "MAX_PLAYERS",
    "POLICY_SIZE",
    "POLICY_SIZE_8x8",
    "POLICY_SIZE_19x19",
    "POLICY_SIZE_HEX8",
    "P_HEX",
    "HEX_BOARD_SIZE",
    "HEX_MAX_DIST",
    "HEX_DIRS",
    "NUM_HEX_DIRS",
    "HEX_PLACEMENT_SPAN",
    "HEX_MOVEMENT_BASE",
    "HEX_MOVEMENT_SPAN",
    "HEX_SPECIAL_BASE",
    "NUM_SQUARE_DIRS",
    "NUM_LINE_DIRS",
    "TERRITORY_SIZE_BUCKETS",
    "TERRITORY_MAX_PLAYERS",
    "BOARD_POLICY_SIZES",
    "BOARD_SPATIAL_SIZES",
    # Square8 layout constants
    "SQUARE8_PLACEMENT_SPAN",
    "SQUARE8_MOVEMENT_BASE",
    "SQUARE8_LINE_FORM_BASE",
    "SQUARE8_TERRITORY_CLAIM_BASE",
    "SQUARE8_SKIP_PLACEMENT_IDX",
    "SQUARE8_SWAP_SIDES_IDX",
    # Hex8 constants
    "HEX8_BOARD_SIZE",
    # Dataclasses
    "DecodedPolicyIndex",
    # Functions
    "get_policy_size_for_board",
    "get_spatial_size_for_board",
    "encode_move_for_board",
    "decode_move_for_board",
    "transform_policy_index_square",
    "create_hex_mask",
    "get_memory_tier",
    "create_model_for_board",
    "get_model_config_for_board",
    # Building blocks
    "ResidualBlock",
    "SEResidualBlock",
    "AttentionResidualBlock",
    # CNN architectures
    "RingRiftCNN_v2",
    "RingRiftCNN_v2_Lite",
    "RingRiftCNN_v3",
    "RingRiftCNN_v3_Lite",
    "RingRiftCNN_v4",
    # Hex architectures
    "ActionEncoderHex",
    "HexNeuralNet_v2",
    "HexNeuralNet_v2_Lite",
    "HexNeuralNet_v3",
    "HexNeuralNet_v3_Lite",
    # Main AI class
    "NeuralNetAI",
    # Loss functions
    "multi_player_value_loss",
    "rank_distribution_loss",
    "ranks_from_game_result",
    # Cache functions
    "clear_model_cache",
    "get_cached_model_count",
]
