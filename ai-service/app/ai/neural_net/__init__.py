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

# Constants - fully migrated to constants.py (Phase 1 complete)
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

# Hex encoding - migrated to hex_encoding.py (Phase 2)
from app.ai.neural_net.hex_encoding import (
    ActionEncoderHex,
    _from_canonical_xy,
    _infer_board_size,
    _pos_from_key,
    _to_canonical_xy,
)

# Square encoding - migrated to square_encoding.py (Phase 2)
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

# Square architectures - migrated to square_architectures.py (Phase 2)
# NOTE: RingRiftCNN_v5 (v5.0.0) was removed Dec 2025 - use RingRiftCNN_v5_Heavy instead
from app.ai.neural_net.square_architectures import (
    RingRiftCNN_v2,
    RingRiftCNN_v2_Lite,
    RingRiftCNN_v3,
    RingRiftCNN_v3_Flat,  # V3 with flat policy (training compatible, Dec 2025)
    RingRiftCNN_v3_Lite,
    RingRiftCNN_v4,
)

# Hex architectures - migrated to hex_architectures.py (Phase 2)
from app.ai.neural_net.hex_architectures import (
    HexNeuralNet_v2,
    HexNeuralNet_v2_Lite,
    HexNeuralNet_v3,
    HexNeuralNet_v3_Flat,  # V3 with flat policy (training compatible, Dec 2025)
    HexNeuralNet_v3_Lite,
    HexNeuralNet_v4,
)

# V5 Heavy architectures - Maximum strength with all features (December 2025)
from app.ai.neural_net.v5_heavy import (
    HeuristicEncoder,
    HexNeuralNet_v5_Heavy,
    RingRiftCNN_v5_Heavy,
    create_v5_heavy_model,
    NUM_HEURISTIC_FEATURES,
)

# Graph encoding - extracted from archive/cage_network.py (December 2025)
# Enables GNN-based position evaluation
from app.ai.neural_net.graph_encoding import (
    NODE_FEATURE_IDX,
    EDGE_ATTR_IDX,
    board_to_graph,
    board_to_graph_hex,
)

# Loss functions - migrated to neural_losses.py (Phase 3)
from app.ai.neural_losses import (
    multi_player_value_loss,
    rank_distribution_loss,
    ranks_from_game_result,
)

# Model factory - migrated to model_factory.py (Phase 4)
from app.ai.neural_net.model_factory import (
    create_model_for_board,
    get_memory_tier,
    get_model_config_for_board,
)

# Canonical move encoding (migrated from _neural_net_legacy.py)
from app.ai.canonical_move_encoding import (
    decode_move_for_board,
    encode_move_for_board,
    get_encoder_for_board,
)

# Unified neural net factory for tournament daemon
# Renamed from neural_net.py to unified_factory.py to avoid package name conflict
from app.ai.unified_factory import UnifiedNeuralNetFactory

# Classes and functions still in legacy module (to be migrated in later phases)
# NOTE: Moved to archive/ December 2025 - scheduled for removal Q1 2026
# Symlinked at app/ai/_neural_net_legacy.py for compatibility
# Suppress deprecation warning for backwards-compatible re-exports
import warnings as _w
with _w.catch_warnings():
    _w.filterwarnings("ignore", category=DeprecationWarning)
    from app.ai._neural_net_legacy import (
        _MODEL_CACHE,
        # Main AI class
        NeuralNetAI,
        # Cache functions
        clear_model_cache,
        create_hex_mask,
        get_cached_model_count,
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
