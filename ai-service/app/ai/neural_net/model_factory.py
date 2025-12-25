"""Model factory for creating board-specific neural network models.

This module provides factory functions to instantiate the appropriate
neural network architecture based on board type and memory tier.

Usage:
    from app.ai.neural_net.model_factory import create_model_for_board

    # Create model for 8x8 board
    model = create_model_for_board(BoardType.SQUARE8)

    # Create model with specific memory tier
    model = create_model_for_board(BoardType.HEXAGONAL, memory_tier="v3-high")

    # Get configuration without creating model
    config = get_model_config_for_board(BoardType.SQUARE19)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch.nn as nn

from app.models import BoardType

from .constants import get_policy_size_for_board, get_spatial_size_for_board
from .hex_architectures import (
    HexNeuralNet_v2,
    HexNeuralNet_v2_Lite,
    HexNeuralNet_v3,
    HexNeuralNet_v3_Lite,
)
from .square_architectures import (
    RingRiftCNN_v2,
    RingRiftCNN_v2_Lite,
    RingRiftCNN_v3,
    RingRiftCNN_v3_Lite,
    RingRiftCNN_v4,
)

# GNN imports (optional - requires PyTorch Geometric)
try:
    from .gnn_policy import GNNPolicyNet, HAS_PYG as HAS_GNN_POLICY
    from .hybrid_cnn_gnn import HybridPolicyNet, HAS_PYG as HAS_HYBRID
    HAS_GNN = HAS_GNN_POLICY or HAS_HYBRID
except ImportError:
    HAS_GNN = False
    GNNPolicyNet = None  # type: ignore
    HybridPolicyNet = None  # type: ignore

logger = logging.getLogger(__name__)

__all__ = [
    "create_model_for_board",
    "get_memory_tier",
    "get_model_config_for_board",
]

# Valid memory tier options
VALID_MEMORY_TIERS = ("high", "low", "v3-high", "v3-low", "v4", "gnn", "hybrid")


def get_memory_tier() -> str:
    """Get the memory tier configuration from environment variable.

    The memory tier controls which model variant to use:
    - "high" (default, 96GB target): Full-capacity v2 models
    - "low" (48GB target): Memory-efficient v2-lite models
    - "v3-high": V3 models with spatial policy heads
    - "v3-low": V3-lite models with spatial policy heads
    - "v4": V4 NAS-optimized models with attention (square boards only)
    - "gnn": Pure Graph Neural Network (GNNPolicyNet, ~255K params)
    - "hybrid": CNN-GNN hybrid architecture (HybridPolicyNet, ~15.5M params)

    Returns:
        One of "high", "low", "v3-high", "v3-low", "v4", "gnn", or "hybrid".
    """
    tier = os.environ.get("RINGRIFT_NN_MEMORY_TIER", "high").lower()
    if tier not in VALID_MEMORY_TIERS:
        logger.warning(f"Unknown memory tier '{tier}', defaulting to 'high'")
        return "high"
    return tier


def create_model_for_board(
    board_type: BoardType,
    in_channels: int = 14,
    global_features: int = 20,
    num_res_blocks: int | None = None,
    num_filters: int | None = None,
    history_length: int = 3,
    memory_tier: str | None = None,
    model_class: str | None = None,
    num_players: int = 4,
    policy_size: int | None = None,
    **_: Any,
) -> nn.Module:
    """Create a neural network model optimized for a specific board type.

    This factory function instantiates the correct model architecture with
    board-specific policy head sizes. All models are CUDA and MPS compatible.

    Parameters
    ----------
    board_type : BoardType
        The board type (SQUARE8, SQUARE19, HEX8, or HEXAGONAL).
    in_channels : int
        Number of input feature channels per frame (default 14).
    global_features : int
        Number of global feature dimensions (default 20).
    num_res_blocks : int, optional
        Number of residual blocks in the backbone (default depends on tier).
    num_filters : int, optional
        Number of convolutional filters (default depends on tier).
    history_length : int
        Number of historical frames to stack (default 3).
    memory_tier : str, optional
        Memory tier override. Valid values:
        - "high" (default, 96GB target): V2 models with GAP→FC policy heads
        - "low" (48GB target): V2-lite models with reduced capacity
        - "v3-high": V3 models with spatial policy heads
        - "v3-low": V3-lite models with spatial policy heads
        - "v4": V4 NAS-optimized architecture with attention (square boards only)
        - "gnn": Pure Graph Neural Network (requires PyTorch Geometric)
        - "hybrid": CNN-GNN hybrid architecture (requires PyTorch Geometric)
        If None, reads from RINGRIFT_NN_MEMORY_TIER environment variable.
    num_players : int
        Number of players (default 4).
    policy_size : int, optional
        Override policy head size. If None, uses board-specific default.

    Returns
    -------
    nn.Module
        A model instance configured for the specified board type.

    Examples
    --------
    >>> model_8x8 = create_model_for_board(BoardType.SQUARE8)
    >>> model_hex = create_model_for_board(BoardType.HEXAGONAL, memory_tier="low")
    """
    # Get board-specific parameters
    board_size = get_spatial_size_for_board(board_type)
    if policy_size is None:
        policy_size = get_policy_size_for_board(board_type)

    # Determine memory tier
    tier = memory_tier if memory_tier is not None else get_memory_tier()

    # Calculate total input channels with history
    total_in_channels = in_channels * (history_length + 1)

    # Check for GNN tiers first (board-agnostic)
    if tier in ("gnn", "hybrid"):
        return _create_gnn_model(
            tier=tier,
            board_type=board_type,
            in_channels=total_in_channels,
            board_size=board_size,
            policy_size=policy_size,
            num_players=num_players,
        )

    # Create model based on board type and memory tier
    if board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
        # Compute hex_radius from board_type: HEX8 has radius 4, HEXAGONAL has radius 12
        hex_radius = 4 if board_type == BoardType.HEX8 else 12
        return _create_hex_model(
            tier=tier,
            in_channels=total_in_channels,
            global_features=global_features,
            num_res_blocks=num_res_blocks,
            num_filters=num_filters,
            board_size=board_size,
            hex_radius=hex_radius,
            policy_size=policy_size,
            num_players=num_players,
        )
    else:
        return _create_square_model(
            tier=tier,
            board_size=board_size,
            in_channels=in_channels,
            global_features=global_features,
            num_res_blocks=num_res_blocks,
            num_filters=num_filters,
            history_length=history_length,
            policy_size=policy_size,
        )


def _create_hex_model(
    tier: str,
    in_channels: int,
    global_features: int,
    num_res_blocks: int | None,
    num_filters: int | None,
    board_size: int,
    hex_radius: int,
    policy_size: int,
    num_players: int,
) -> nn.Module:
    """Create a hexagonal board model based on memory tier.

    Args:
        tier: Memory tier (v3-high, v3-low, high, low)
        in_channels: Number of input channels
        global_features: Number of global feature dimensions
        num_res_blocks: Number of residual blocks (or None for default)
        num_filters: Number of convolutional filters (or None for default)
        board_size: Spatial size of the board (9 for HEX8, 25 for HEXAGONAL)
        hex_radius: Hexagonal grid radius (4 for HEX8, 12 for HEXAGONAL)
        policy_size: Size of the policy output
        num_players: Number of players
    """
    if tier == "v4":
        raise ValueError(
            "V4 architecture is not yet available for hexagonal boards. "
            "Use 'v3-high', 'v3-low', 'high', or 'low' instead."
        )

    if tier == "v3-high":
        return HexNeuralNet_v3(
            in_channels=in_channels,
            global_features=global_features,
            num_res_blocks=num_res_blocks or 12,
            num_filters=num_filters or 192,
            board_size=board_size,
            hex_radius=hex_radius,
            policy_size=policy_size,
            num_players=num_players,
        )
    elif tier == "v3-low":
        return HexNeuralNet_v3_Lite(
            in_channels=in_channels,
            global_features=global_features,
            num_res_blocks=num_res_blocks or 6,
            num_filters=num_filters or 96,
            board_size=board_size,
            hex_radius=hex_radius,
            policy_size=policy_size,
            num_players=num_players,
        )
    elif tier == "high":
        return HexNeuralNet_v2(
            in_channels=in_channels,
            global_features=global_features,
            num_res_blocks=num_res_blocks or 12,
            num_filters=num_filters or 192,
            board_size=board_size,
            hex_radius=hex_radius,
            policy_size=policy_size,
            num_players=num_players,
        )
    else:  # low tier
        return HexNeuralNet_v2_Lite(
            in_channels=in_channels,
            global_features=global_features,
            num_res_blocks=num_res_blocks or 6,
            num_filters=num_filters or 96,
            board_size=board_size,
            hex_radius=hex_radius,
            policy_size=policy_size,
            num_players=num_players,
        )


def _create_gnn_model(
    tier: str,
    board_type: BoardType,
    in_channels: int,
    board_size: int,
    policy_size: int,
    num_players: int,
) -> nn.Module:
    """Create a GNN-based model (requires PyTorch Geometric).

    Args:
        tier: Memory tier ("gnn" or "hybrid")
        board_type: The board type
        in_channels: Number of input channels
        board_size: Spatial size of the board
        policy_size: Size of the policy output
        num_players: Number of players

    Raises:
        ImportError: If PyTorch Geometric is not installed
        ValueError: For unsupported tier values
    """
    if not HAS_GNN:
        raise ImportError(
            "GNN models require PyTorch Geometric. Install with: "
            "pip install torch-geometric torch-scatter torch-sparse"
        )

    is_hex = board_type in (BoardType.HEXAGONAL, BoardType.HEX8)

    if tier == "gnn":
        if GNNPolicyNet is None:
            raise ImportError("GNNPolicyNet not available - PyTorch Geometric required")
        return GNNPolicyNet(
            node_feature_dim=in_channels,  # Must match encoder output dimension
            hidden_dim=128,
            num_layers=6,
            action_space_size=policy_size,
            num_players=num_players,
        )
    elif tier == "hybrid":
        if HybridPolicyNet is None:
            raise ImportError("HybridPolicyNet not available - PyTorch Geometric required")
        return HybridPolicyNet(
            in_channels=in_channels,
            hidden_channels=128,
            cnn_blocks=6,
            gnn_layers=3,
            board_size=board_size,
            action_space_size=policy_size,
            num_players=num_players,
            is_hex=is_hex,
        )
    else:
        raise ValueError(f"Unknown GNN tier: {tier}")


def _create_square_model(
    tier: str,
    board_size: int,
    in_channels: int,
    global_features: int,
    num_res_blocks: int | None,
    num_filters: int | None,
    history_length: int,
    policy_size: int,
) -> nn.Module:
    """Create a square board model based on memory tier."""
    if tier == "v4":
        return RingRiftCNN_v4(
            board_size=board_size,
            in_channels=in_channels,
            global_features=global_features,
            num_res_blocks=num_res_blocks or 13,  # NAS optimal
            num_filters=num_filters or 128,  # NAS optimal
            num_attention_heads=4,  # NAS optimal
            dropout=0.08,  # NAS optimal
            initial_kernel_size=5,  # NAS optimal
            history_length=history_length,
            policy_size=policy_size,
        )
    elif tier == "v3-high":
        return RingRiftCNN_v3(
            board_size=board_size,
            in_channels=in_channels,
            global_features=global_features,
            num_res_blocks=num_res_blocks or 12,
            num_filters=num_filters or 192,
            history_length=history_length,
            policy_size=policy_size,
        )
    elif tier == "v3-low":
        return RingRiftCNN_v3_Lite(
            board_size=board_size,
            in_channels=in_channels,
            global_features=global_features,
            num_res_blocks=num_res_blocks or 6,
            num_filters=num_filters or 96,
            history_length=history_length,
            policy_size=policy_size,
        )
    elif tier == "high":
        return RingRiftCNN_v2(
            board_size=board_size,
            in_channels=in_channels,
            global_features=global_features,
            num_res_blocks=num_res_blocks or 12,
            num_filters=num_filters or 192,
            history_length=history_length,
            policy_size=policy_size,
        )
    else:  # low tier
        return RingRiftCNN_v2_Lite(
            board_size=board_size,
            in_channels=in_channels,
            global_features=global_features,
            num_res_blocks=num_res_blocks or 6,
            num_filters=num_filters or 96,
            history_length=history_length,
            policy_size=policy_size,
        )


def get_model_config_for_board(
    board_type: BoardType,
    memory_tier: str | None = None,
) -> dict[str, Any]:
    """Get recommended model configuration for a specific board type.

    Returns a dictionary of hyperparameters optimized for the board type,
    including recommended residual block count and filter count based on
    the complexity of the action space and memory tier.

    Parameters
    ----------
    board_type : BoardType
        The board type to get configuration for.
    memory_tier : str, optional
        Memory tier override: "high", "low", "v3-high", "v3-low", or "v4".
        If None, reads from RINGRIFT_NN_MEMORY_TIER environment variable.

    Returns
    -------
    dict[str, Any]
        Configuration dictionary with keys:
        - board_size: Spatial dimension of the board
        - policy_size: Action space size
        - num_res_blocks: Recommended residual block count
        - num_filters: Recommended filter count
        - recommended_model: Which model class to use
        - memory_tier: Active memory tier
        - estimated_params_m: Estimated parameter count in millions
    """
    tier = memory_tier if memory_tier is not None else get_memory_tier()

    config: dict[str, Any] = {
        "board_size": get_spatial_size_for_board(board_type),
        "policy_size": get_policy_size_for_board(board_type),
        "memory_tier": tier,
    }

    # Get tier-specific configuration
    tier_config = _get_tier_config(board_type, tier)
    config.update(tier_config)

    return config


def _get_tier_config(board_type: BoardType, tier: str) -> dict[str, Any]:
    """Get configuration for a specific board type and memory tier."""
    is_hex = board_type in (BoardType.HEXAGONAL, BoardType.HEX8)
    is_hex8 = board_type == BoardType.HEX8
    is_sq19 = board_type == BoardType.SQUARE19

    # Configuration lookup table
    configs = {
        "v3-high": {
            "num_res_blocks": 12,
            "num_filters": 192,
            "estimated_params_m": 7.0 if is_hex8 else 8.2 if is_hex else 7.0,
            "recommended_model": "HexNeuralNet_v3" if is_hex else "RingRiftCNN_v3",
            "description": f"V3 spatial policy model for 96GB systems (~{7 if is_hex8 else 8 if is_hex else 7}M params)",
        },
        "v3-low": {
            "num_res_blocks": 6,
            "num_filters": 96,
            "estimated_params_m": 1.8 if is_hex8 else 2.1 if is_hex else 1.8,
            "recommended_model": "HexNeuralNet_v3_Lite" if is_hex else "RingRiftCNN_v3_Lite",
            "description": "V3 spatial policy model for 48GB systems (~2M params)",
        },
        "high": {
            "num_res_blocks": 12,
            "num_filters": 192,
            "estimated_params_m": 34.0 if is_hex8 else 43.4 if is_hex else 34.0,
            "recommended_model": "HexNeuralNet_v2" if is_hex else "RingRiftCNN_v2",
            "description": f"High-capacity model for 96GB systems (~{34 if is_hex8 else 43 if is_hex else 34}M params)",
        },
        "low": {
            "num_res_blocks": 6,
            "num_filters": 96,
            "estimated_params_m": 14.0 if is_hex8 else 18.7 if is_hex else 14.3 if is_sq19 else 14.0,
            "recommended_model": "HexNeuralNet_v2_Lite" if is_hex else "RingRiftCNN_v2_Lite",
            "description": f"Memory-efficient model for 48GB systems (~{14 if is_hex8 else 19 if is_hex else 14}M params)",
        },
        "v4": {
            "num_res_blocks": 13,
            "num_filters": 128,
            "estimated_params_m": 8.5,
            "recommended_model": "RingRiftCNN_v4",
            "description": "V4 NAS-optimized model with attention (~8.5M params)",
        },
        "gnn": {
            "num_res_blocks": 0,  # Not applicable for GNN
            "num_filters": 128,  # hidden_dim
            "estimated_params_m": 0.9,
            "recommended_model": "GNNPolicyNet",
            "description": "Pure GNN model (~900K params, requires PyTorch Geometric)",
            "requires_pyg": True,
        },
        "hybrid": {
            "num_res_blocks": 6,  # cnn_blocks
            "num_filters": 128,  # hidden_channels
            "estimated_params_m": 17.3,
            "recommended_model": "HybridPolicyNet",
            "description": "CNN-GNN hybrid model (~17M params, requires PyTorch Geometric)",
            "requires_pyg": True,
        },
    }

    return configs.get(tier, configs["high"])
