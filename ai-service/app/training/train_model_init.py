"""
Model initialization utilities for training.

Extracted from train.py (December 2025) to reduce module size.
Handles model creation, weight loading, and transfer learning setup.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .train_config import TrainConfig

logger = logging.getLogger(__name__)

# Maximum players supported
MAX_PLAYERS = 4


@dataclass
class ModelInitConfig:
    """Configuration for model initialization."""

    board_size: int
    policy_size: int
    num_players: int
    multi_player: bool
    model_version: str
    model_type: str  # "cnn", "gnn", "hybrid"
    num_res_blocks: int | None
    num_filters: int | None
    dropout: float
    history_length: int
    board_type: str
    # Hex-specific
    use_hex_model: bool = False
    hex_in_channels: int | None = None
    hex_radius: int | None = None
    use_hex_v3: bool = False
    use_hex_v4: bool = False
    # Feature version for compatibility
    feature_version: int = 2


@dataclass
class ModelInitResult:
    """Result of model initialization."""

    model: nn.Module
    effective_blocks: int
    effective_filters: int
    model_name: str


def determine_architecture_size(
    model_version: str,
    use_hex_v4: bool,
    use_hex_model: bool,
    num_res_blocks: int | None,
    num_filters: int | None,
) -> tuple[int, int]:
    """Determine the effective number of residual blocks and filters.

    Returns:
        (effective_blocks, effective_filters)
    """
    if use_hex_v4 or model_version == 'v4':
        # NAS optimal for v4
        effective_blocks = num_res_blocks if num_res_blocks is not None else 13
        effective_filters = num_filters if num_filters is not None else 128
    elif model_version == 'v3' or use_hex_model:
        effective_blocks = num_res_blocks if num_res_blocks is not None else 12
        effective_filters = num_filters if num_filters is not None else 192
    else:
        # v2 defaults
        effective_blocks = num_res_blocks if num_res_blocks is not None else 6
        effective_filters = num_filters if num_filters is not None else 96

    return effective_blocks, effective_filters


def create_model(config: ModelInitConfig) -> ModelInitResult:
    """Create a neural network model based on configuration.

    Args:
        config: Model initialization configuration

    Returns:
        ModelInitResult with model and metadata
    """
    from app.ai.neural_net import (
        HexNeuralNet_v2,
        HexNeuralNet_v3,
        HexNeuralNet_v4,
        RingRiftCNN_v2,
        RingRiftCNN_v3,
    )

    effective_blocks, effective_filters = determine_architecture_size(
        config.model_version,
        config.use_hex_v4,
        config.use_hex_model,
        config.num_res_blocks,
        config.num_filters,
    )

    model: nn.Module
    model_name: str

    # GNN/Hybrid models use model_factory
    if config.model_type in ("gnn", "hybrid"):
        from app.ai.neural_net.model_factory import create_model_for_board, HAS_GNN

        if not HAS_GNN:
            raise ImportError(
                f"Model type '{config.model_type}' requires PyTorch Geometric. "
                "Install with: pip install torch-geometric torch-scatter torch-sparse"
            )
        gnn_num_players = MAX_PLAYERS if config.multi_player else config.num_players
        model = create_model_for_board(
            board_type=config.board_type,
            memory_tier=config.model_type,
            num_players=gnn_num_players,
        )
        model_name = f"{config.model_type.upper()}"

    elif config.use_hex_v4:
        hex_num_players = MAX_PLAYERS if config.multi_player else config.num_players
        model = HexNeuralNet_v4(
            in_channels=config.hex_in_channels or 64,
            global_features=20,
            num_res_blocks=effective_blocks,
            num_filters=effective_filters,
            board_size=config.board_size,
            hex_radius=config.hex_radius or (config.board_size // 2),
            policy_size=config.policy_size,
            num_players=hex_num_players,
        )
        model_name = "HexNeuralNet_v4"

    elif config.use_hex_v3:
        hex_num_players = MAX_PLAYERS if config.multi_player else config.num_players
        model = HexNeuralNet_v3(
            in_channels=config.hex_in_channels or 64,
            global_features=20,
            num_res_blocks=effective_blocks,
            num_filters=effective_filters,
            board_size=config.board_size,
            hex_radius=config.hex_radius or (config.board_size // 2),
            policy_size=config.policy_size,
            num_players=hex_num_players,
        )
        model_name = "HexNeuralNet_v3"

    elif config.use_hex_model:
        hex_num_players = MAX_PLAYERS if config.multi_player else config.num_players
        model = HexNeuralNet_v2(
            in_channels=config.hex_in_channels or 40,
            global_features=20,
            num_res_blocks=effective_blocks,
            num_filters=effective_filters,
            board_size=config.board_size,
            hex_radius=config.hex_radius or (config.board_size // 2),
            policy_size=config.policy_size,
            num_players=hex_num_players,
        )
        model_name = "HexNeuralNet_v2"

    elif config.model_version == 'v4':
        from app.ai.neural_net import RingRiftCNN_v4

        v4_num_players = MAX_PLAYERS if config.multi_player else config.num_players
        model = RingRiftCNN_v4(
            board_size=config.board_size,
            in_channels=14,
            global_features=20,
            history_length=config.history_length,
            policy_size=config.policy_size,
            num_players=v4_num_players,
            num_res_blocks=effective_blocks,
            num_filters=effective_filters,
            num_attention_heads=4,
            dropout=config.dropout,
            initial_kernel_size=5,
        )
        model_name = "RingRiftCNN_v4"

    elif config.model_version == 'v3':
        v3_num_players = MAX_PLAYERS if config.multi_player else config.num_players
        model = RingRiftCNN_v3(
            board_size=config.board_size,
            in_channels=14,
            global_features=20,
            history_length=config.history_length,
            policy_size=config.policy_size,
            num_players=v3_num_players,
            num_res_blocks=effective_blocks,
            num_filters=effective_filters,
        )
        model_name = "RingRiftCNN_v3"

    elif config.multi_player:
        mp_num_players = config.num_players if config.num_players in (2, 3, 4) else MAX_PLAYERS
        model = RingRiftCNN_v2(
            board_size=config.board_size,
            in_channels=14,
            global_features=20,
            history_length=config.history_length,
            policy_size=config.policy_size,
            num_res_blocks=effective_blocks,
            num_filters=effective_filters,
            num_players=mp_num_players,
        )
        model_name = f"RingRiftCNN_v2_{mp_num_players}p"

    else:
        v2_num_players = config.num_players if config.num_players in (2, 3, 4) else 2
        model = RingRiftCNN_v2(
            board_size=config.board_size,
            in_channels=14,
            global_features=20,
            history_length=config.history_length,
            policy_size=config.policy_size,
            num_res_blocks=effective_blocks,
            num_filters=effective_filters,
            num_players=v2_num_players,
        )
        model_name = "RingRiftCNN_v2"

    # Set feature version for compatibility
    with contextlib.suppress(Exception):
        model.feature_version = config.feature_version

    return ModelInitResult(
        model=model,
        effective_blocks=effective_blocks,
        effective_filters=effective_filters,
        model_name=model_name,
    )


def load_initial_weights(
    model: nn.Module,
    weights_path: str,
    strict: bool = False,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load initial weights into model with optional strict matching.

    Args:
        model: Model to load weights into
        weights_path: Path to weights file (.pth or .pt)
        strict: If True, require exact key match
        device: Device to load weights to

    Returns:
        Dict with 'missing_keys' and 'unexpected_keys' lists
    """
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    # Handle wrapped state dicts (e.g., from checkpoints)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Remove 'module.' prefix if present (from DDP)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    # Load with strict mode
    if strict:
        model.load_state_dict(cleaned_state_dict, strict=True)
        return {"missing_keys": [], "unexpected_keys": []}
    else:
        result = model.load_state_dict(cleaned_state_dict, strict=False)
        return {
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }


def setup_transfer_learning(
    model: nn.Module,
    source_num_players: int,
    target_num_players: int,
) -> None:
    """Resize value head for transfer learning between player counts.

    Args:
        model: Model with value head to resize
        source_num_players: Number of players in source model
        target_num_players: Number of players in target training
    """
    if source_num_players == target_num_players:
        return

    logger.info(
        f"Transfer learning: resizing value head from {source_num_players} to {target_num_players} players"
    )

    # Find and resize the final value head layer
    if hasattr(model, "value_head"):
        value_head = model.value_head
        if isinstance(value_head, nn.Sequential):
            # Find the last linear layer
            for i in range(len(value_head) - 1, -1, -1):
                if isinstance(value_head[i], nn.Linear):
                    old_layer = value_head[i]
                    new_layer = nn.Linear(
                        old_layer.in_features,
                        target_num_players,
                        bias=old_layer.bias is not None,
                    )
                    # Initialize with Xavier
                    nn.init.xavier_uniform_(new_layer.weight)
                    if new_layer.bias is not None:
                        nn.init.zeros_(new_layer.bias)
                    value_head[i] = new_layer
                    logger.info(f"Replaced value head layer {i}: {old_layer} -> {new_layer}")
                    break
        elif isinstance(value_head, nn.Linear):
            new_layer = nn.Linear(
                value_head.in_features,
                target_num_players,
                bias=value_head.bias is not None,
            )
            nn.init.xavier_uniform_(new_layer.weight)
            if new_layer.bias is not None:
                nn.init.zeros_(new_layer.bias)
            model.value_head = new_layer
            logger.info(f"Replaced value head: {value_head} -> {new_layer}")


def freeze_policy_head(model: nn.Module) -> list[nn.Parameter]:
    """Freeze policy head parameters, return value head parameters for training.

    Args:
        model: Model with policy_head attribute

    Returns:
        List of value head parameters (unfrozen) for optimizer
    """
    value_head_params = []

    for name, param in model.named_parameters():
        if "value" in name.lower():
            param.requires_grad = True
            value_head_params.append(param)
            logger.info(f"[freeze_policy] Unfreezing: {name}")
        else:
            param.requires_grad = False

    if not value_head_params:
        logger.warning(
            "[freeze_policy] No value head parameters found. "
            "All parameters will be frozen."
        )

    logger.info(f"[freeze_policy] Training only {len(value_head_params)} value head parameters")
    return value_head_params
