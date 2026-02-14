"""Model factory for RingRift training pipeline.

Extracted from train.py (lines 2532-2858) to reduce train_model() complexity.
Contains the model creation if/elif chain that selects the appropriate neural
network architecture based on board type, model version, and other parameters.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)


def create_training_model(
    *,
    config: Any,  # TrainConfig
    model_version: str,
    model_type: str,
    board_size: int,
    policy_size: int,
    num_players: int,
    hex_in_channels: int,
    hex_radius: int,
    hex_num_players: int,
    use_hex_model: bool,
    use_hex_v3: bool,
    use_hex_v4: bool,
    use_hex_v5: bool,
    use_hex_v5_large: bool,
    detected_num_heuristics: int | None,
    effective_blocks: int,
    effective_filters: int,
    multi_player: bool,
    dropout: float,
    config_feature_version: int,
    distributed: bool = False,
    is_main: bool = True,
    # Import references - model classes
    HexNeuralNet_v2: Any = None,
    HexNeuralNet_v3: Any = None,
    HexNeuralNet_v3_Flat: Any = None,
    HexNeuralNet_v4: Any = None,
    HexNeuralNet_v5_Heavy: Any = None,
    RingRiftCNN_v2: Any = None,
    RingRiftCNN_v3: Any = None,
    RingRiftCNN_v3_Flat: Any = None,
    MAX_PLAYERS: int = 4,
) -> nn.Module:
    """Create and return the appropriate neural network model for training.

    Selects the model architecture based on board type, model version, and
    other configuration parameters. Supports:
    - GNN/Hybrid models (via model_factory)
    - Hex V5 Heavy (with heuristic features)
    - Hex V4 (NAS-optimized attention)
    - Hex V3 (spatial and flat policy heads)
    - Hex V5 Heavy Large/XL
    - Hex V2 (standard hex)
    - Square V3 Spatial/Flat
    - Square V4 (NAS-optimized)
    - Square V5/V5-Heavy
    - Square V5-Heavy-Large/XL
    - Multi-player V2
    - Default V2

    Args:
        config: Training configuration.
        model_version: Model version string (v2, v3, v4, v5-heavy, etc.).
        model_type: Model type (cnn, gnn, hybrid).
        board_size: Spatial board size for CNN.
        policy_size: Policy head output size.
        num_players: Number of players.
        hex_in_channels: Input channels for hex models.
        hex_radius: Hex board radius.
        hex_num_players: Number of players for hex models.
        use_hex_model: Whether to use hex neural net.
        use_hex_v3: Whether to use hex V3.
        use_hex_v4: Whether to use hex V4.
        use_hex_v5: Whether to use hex V5.
        use_hex_v5_large: Whether to use hex V5 large.
        detected_num_heuristics: Detected heuristic feature count from NPZ.
        effective_blocks: Number of residual blocks.
        effective_filters: Number of CNN filters.
        multi_player: Whether multi-player mode is enabled.
        dropout: Dropout rate.
        config_feature_version: Feature version from config.
        distributed: Whether distributed training is enabled.
        is_main: Whether this is the main process.
        HexNeuralNet_v2: HexNeuralNet_v2 class reference.
        HexNeuralNet_v3: HexNeuralNet_v3 class reference.
        HexNeuralNet_v3_Flat: HexNeuralNet_v3_Flat class reference.
        HexNeuralNet_v4: HexNeuralNet_v4 class reference.
        HexNeuralNet_v5_Heavy: HexNeuralNet_v5_Heavy class reference.
        RingRiftCNN_v2: RingRiftCNN_v2 class reference.
        RingRiftCNN_v3: RingRiftCNN_v3 class reference.
        RingRiftCNN_v3_Flat: RingRiftCNN_v3_Flat class reference.
        MAX_PLAYERS: Maximum player count constant.

    Returns:
        Initialized neural network model (not yet moved to device).
    """
    model: nn.Module

    # GNN/Hybrid models use model_factory for unified creation
    if model_type in ("gnn", "hybrid"):
        from app.ai.neural_net.model_factory import create_model_for_board, HAS_GNN
        if not HAS_GNN:
            raise ImportError(
                f"Model type '{model_type}' requires PyTorch Geometric. "
                "Install with: pip install torch-geometric torch-scatter torch-sparse"
            )
        gnn_num_players = MAX_PLAYERS if multi_player else num_players
        model = create_model_for_board(
            board_type=config.board_type,
            memory_tier=model_type,
            num_players=gnn_num_players,
        )
        if not distributed or is_main:
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(
                f"Initialized {model_type.upper()} model for {config.board_type.name}: "
                f"{param_count:,} parameters"
            )
    elif use_hex_v5:
        # HexNeuralNet_v5_Heavy for hexagonal boards with all features
        # V5 uses 16 base channels * (history_length + 1) frames = 64 channels
        from app.ai.neural_net.v5_heavy import NUM_HEURISTIC_FEATURES_FAST
        v5_filters = effective_filters
        use_gnn = model_version in ('v5-gnn',)
        # Use detected heuristic count from NPZ, or default to fast mode (21)
        hex_v5_num_heuristics = detected_num_heuristics if detected_num_heuristics else NUM_HEURISTIC_FEATURES_FAST
        model = HexNeuralNet_v5_Heavy(
            board_size=board_size,
            hex_radius=hex_radius,
            in_channels=hex_in_channels,
            global_features=20,
            num_filters=v5_filters,
            policy_size=policy_size,
            num_players=hex_num_players,
            num_heuristics=hex_v5_num_heuristics,  # Auto-detected from NPZ
            use_gnn=use_gnn,
            dropout=dropout,
        )
    elif use_hex_v4:
        # HexNeuralNet_v4 for hexagonal boards with NAS-optimized attention
        # V4 uses 16 base channels * (history_length + 1) frames = 64 channels
        # NOTE: Pass policy_size=None so V4 computes it dynamically from board_size.
        # This ensures correct policy_size for any hex board (hex8=4132, hexagonal=91876).
        # Passing explicit policy_size=4500 would cause -1e9 logits outside scatter range.
        model = HexNeuralNet_v4(
            in_channels=hex_in_channels,
            global_features=20,  # V4 encoder provides 20 global features
            num_res_blocks=effective_blocks,
            num_filters=effective_filters,
            board_size=board_size,
            hex_radius=hex_radius,
            policy_size=None,  # Computed dynamically from board_size
            num_players=hex_num_players,
        )
    elif use_hex_v3:
        # HexNeuralNet_v3 for hexagonal boards (Jan 2026)
        # NOTE: Spatial policy heads cause loss explosion (~42 policy loss) due to
        # -1e9 masking creating log_softmax numerical instability. Use flat by default.
        # See hex_architectures.py lines 878-917 for detailed explanation.
        if model_version == 'v3-spatial':
            # V3 with spatial policy heads
            logger.info(
                "Using V3 spatial policy heads (v3-spatial). "
                "Policy size computed dynamically from board_size=%d.", board_size
            )
            model = HexNeuralNet_v3(
                in_channels=hex_in_channels,
                global_features=20,
                num_res_blocks=effective_blocks,
                num_filters=effective_filters,
                board_size=board_size,
                hex_radius=hex_radius,
                # NOTE: Pass policy_size=None so V3 computes it dynamically from board_size.
                policy_size=None,
                num_players=hex_num_players,
            )
        else:
            # V3 with flat policy heads (default, stable)
            model = HexNeuralNet_v3_Flat(
                in_channels=hex_in_channels,
                global_features=20,  # V3 encoder provides 20 global features
                num_res_blocks=effective_blocks,
                num_filters=effective_filters,
                board_size=board_size,
                hex_radius=hex_radius,
                policy_size=policy_size,
                num_players=hex_num_players,
            )
    elif use_hex_v5_large:
        # HexNeuralNet_v5_Heavy with large scaling for hexagonal boards (January 2026)
        from app.ai.neural_net.v5_heavy_large import create_v5_heavy_large
        hex_variant_map = {"v5-heavy-xl": "xl", "v6-xl": "xl"}
        hex_large_variant = hex_variant_map.get(model_version, "large")
        hex_large_num_heuristics = detected_num_heuristics if detected_num_heuristics else 49
        model = create_v5_heavy_large(
            board_type=config.board_type.name.lower(),
            num_players=hex_num_players,
            variant=hex_large_variant,
            num_heuristics=hex_large_num_heuristics,
            dropout=dropout,
            in_channels=hex_in_channels,  # Match training data channels
        )
        if not distributed or is_main:
            param_count = sum(p.numel() for p in model.parameters())
            display_name = f"V5-Heavy-{hex_large_variant.upper()}"
            logger.info(
                f"Initializing {display_name} (hex) model for {config.board_type.name}: "
                f"{param_count:,} parameters, num_players={hex_num_players}, "
                f"heuristics={hex_large_num_heuristics}"
            )
    elif use_hex_model:
        # HexNeuralNet_v2 for hexagonal boards with multi-player support
        # V2 uses 10 base channels * (history_length + 1) frames = 40 channels
        model = HexNeuralNet_v2(
            in_channels=hex_in_channels,
            global_features=20,  # Must match _extract_features() which returns 20 globals
            num_res_blocks=effective_blocks,
            num_filters=effective_filters,
            board_size=board_size,
            hex_radius=hex_radius,
            policy_size=policy_size,
            num_players=hex_num_players,
        )
    elif model_version == 'v3-spatial':
        # V3 with spatial policy heads (BROKEN - use for debugging only)
        v3_num_players = MAX_PLAYERS if multi_player else num_players
        logger.warning(
            "Using V3 spatial policy heads (v3-spatial). This is known to cause "
            "loss explosion (~42 policy loss). Use --model-version v3 for flat heads."
        )
        model = RingRiftCNN_v3(
            board_size=board_size,
            in_channels=14,  # 14 spatial feature channels per frame
            global_features=20,  # Must match _extract_features() which returns 20 globals
            history_length=config.history_length,
            policy_size=policy_size,
            num_players=v3_num_players,
            num_res_blocks=effective_blocks,
            num_filters=effective_filters,
        )
        if not distributed or is_main:
            logger.info(
                f"Initializing RingRiftCNN_v3 (spatial) with board_size={board_size}, "
                f"policy_size={policy_size}, num_players={v3_num_players}, "
                f"blocks={effective_blocks}, filters={effective_filters}"
            )
    elif model_version in ('v3', 'v3-flat'):
        # V3 with flat policy heads (default, stable)
        v3_num_players = MAX_PLAYERS if multi_player else num_players
        model = RingRiftCNN_v3_Flat(
            board_size=board_size,
            in_channels=14,  # 14 spatial feature channels per frame
            global_features=20,  # Must match _extract_features() which returns 20 globals
            history_length=config.history_length,
            policy_size=policy_size,
            num_players=v3_num_players,
            num_res_blocks=effective_blocks,
            num_filters=effective_filters,
        )
        if not distributed or is_main:
            logger.info(
                f"Initializing RingRiftCNN_v3_Flat with board_size={board_size}, "
                f"policy_size={policy_size}, num_players={v3_num_players}, "
                f"blocks={effective_blocks}, filters={effective_filters}"
            )
    elif model_version == 'v4':
        # V4 NAS-optimized architecture with multi-head attention (square boards only)
        from app.ai.neural_net import RingRiftCNN_v4
        v4_num_players = MAX_PLAYERS if multi_player else num_players
        # V4 uses NAS-discovered defaults: 13 blocks, 128 filters, 4-head attention
        v4_blocks = effective_blocks
        v4_filters = effective_filters
        model = RingRiftCNN_v4(
            board_size=board_size,
            in_channels=14,  # 14 spatial feature channels per frame
            global_features=20,  # Must match _extract_features() which returns 20 globals
            history_length=config.history_length,
            policy_size=policy_size,
            num_players=v4_num_players,
            num_res_blocks=v4_blocks,
            num_filters=v4_filters,
            num_attention_heads=4,  # NAS optimal
            dropout=dropout,  # Configurable, default NAS optimal 0.08
            initial_kernel_size=5,  # NAS optimal
        )
        if not distributed or is_main:
            logger.info(
                f"Initializing RingRiftCNN_v4 (NAS) with board_size={board_size}, "
                f"policy_size={policy_size}, num_players={v4_num_players}, "
                f"blocks={v4_blocks}, filters={v4_filters}, attention_heads=4"
            )
    elif model_version in ('v5', 'v5-gnn', 'v5-heavy'):
        # V5 Heavy architecture with all features (December 2025)
        from app.ai.neural_net import RingRiftCNN_v5_Heavy
        from app.ai.neural_net.v5_heavy import NUM_HEURISTIC_FEATURES_FAST
        v5_num_players = MAX_PLAYERS if multi_player else num_players
        v5_filters = effective_filters
        use_gnn = model_version in ('v5-gnn',)
        # Use detected heuristic count from NPZ, or default to fast mode (21)
        v5_num_heuristics = detected_num_heuristics if detected_num_heuristics else NUM_HEURISTIC_FEATURES_FAST
        # January 2026: Fixed in_channels - must be total channels (14 base x 4 frames = 56)
        v5_in_channels = 14 * (config.history_length + 1)  # 14 x 4 = 56
        model = RingRiftCNN_v5_Heavy(
            board_size=board_size,
            in_channels=v5_in_channels,  # Total channels: 14 base x (history + 1) frames
            global_features=20,  # Must match _extract_features() which returns 20 globals
            history_length=config.history_length,
            policy_size=policy_size,
            num_players=v5_num_players,
            num_filters=v5_filters,
            num_heuristics=v5_num_heuristics,  # Auto-detected from NPZ
            use_gnn=use_gnn,
            dropout=dropout,
        )
        if not distributed or is_main:
            heuristic_mode_str = "full (49)" if v5_num_heuristics >= 49 else "fast (21)"
            logger.info(
                f"Initializing RingRiftCNN_v5_Heavy with board_size={board_size}, "
                f"policy_size={policy_size}, num_players={v5_num_players}, "
                f"filters={v5_filters}, use_gnn={use_gnn}, heuristics={heuristic_mode_str}"
            )
    elif model_version in ('v5-heavy-large', 'v5-heavy-xl', 'v6', 'v6-xl'):
        # V5 Heavy Large architecture: Scaled-up model for 2000+ Elo (December 2025)
        from app.ai.neural_net.v5_heavy_large import create_v5_heavy_large
        large_num_players = MAX_PLAYERS if multi_player else num_players
        variant_map = {"v5-heavy-xl": "xl", "v6-xl": "xl"}
        large_variant = variant_map.get(model_version, "large")
        large_num_heuristics = detected_num_heuristics if detected_num_heuristics else 49
        v5_large_in_channels = 14 * (config.history_length + 1)  # 14 x 4 = 56 for history_length=3
        model = create_v5_heavy_large(
            board_type=config.board_type.name.lower(),
            num_players=large_num_players,
            variant=large_variant,
            num_heuristics=large_num_heuristics,
            dropout=dropout,
            in_channels=v5_large_in_channels,
            history_length=config.history_length,
        )
        if not distributed or is_main:
            param_count = sum(p.numel() for p in model.parameters())
            display_name = f"V5-Heavy-{large_variant.upper()}"
            logger.info(
                f"Initializing {display_name} model for {config.board_type.name}: "
                f"{param_count:,} parameters, num_players={large_num_players}, "
                f"heuristics={large_num_heuristics}"
            )
    elif multi_player:
        # Multi-player mode: RingRiftCNN_v2 with per-player value head
        mp_num_players = num_players if num_players in (2, 3, 4) else MAX_PLAYERS
        model = RingRiftCNN_v2(
            board_size=board_size,
            in_channels=14,  # 14 spatial feature channels per frame
            global_features=20,  # Must match _extract_features() which returns 20 globals
            history_length=config.history_length,
            policy_size=policy_size,
            num_res_blocks=effective_blocks,
            num_filters=effective_filters,
            num_players=mp_num_players,  # Use actual player count, not MAX_PLAYERS
        )
        if not distributed or is_main:
            logger.info(
                f"Multi-player mode: RingRiftCNN_v2 with {mp_num_players}-player value head."
            )
    else:
        # RingRiftCNN_v2 for square boards
        v2_num_players = num_players if num_players in (2, 3, 4) else 2
        model = RingRiftCNN_v2(
            board_size=board_size,
            in_channels=14,  # 14 spatial feature channels per frame
            global_features=20,  # Must match _extract_features() which returns 20 globals
            history_length=config.history_length,
            policy_size=policy_size,
            num_res_blocks=effective_blocks,
            num_filters=effective_filters,
            num_players=v2_num_players,
        )

    with contextlib.suppress(Exception):
        model.feature_version = config_feature_version

    return model
