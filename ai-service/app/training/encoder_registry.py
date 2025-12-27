"""Centralized encoder registry for RingRift neural network training.

This module provides a single source of truth for encoder configurations,
ensuring consistency between data encoding and model architecture.

Usage:
    from app.training.encoder_registry import get_encoder_config, EncoderConfig

    config = get_encoder_config(BoardType.HEXAGONAL, model_version="v2")
    print(f"Model expects {config.in_channels} input channels")

    # Validate dataset compatibility
    if dataset_channels != config.in_channels:
        raise ValueError(f"Dataset has {dataset_channels} channels, model expects {config.in_channels}")

Channel configurations:
    - hex_v2: 10 base channels × 4 frames = 40 total (HexNeuralNet_v2)
    - hex_v3: 16 base channels × 4 frames = 64 total (HexNeuralNet_v3)
    - square: 14 base channels × 4 frames = 56 total (PolicyValueNet)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.models import BoardType


@dataclass
class EncoderConfig:
    """Configuration for a specific encoder type.

    Attributes:
        encoder_type: Name of the encoder (e.g., "hex_v2", "hex_v3", "square")
        encoder_class_name: Class name for lazy import (e.g., "HexStateEncoder")
        base_channels: Number of feature channels per frame
        history_length: Number of history frames (default 3)
        in_channels: Total input channels (base × (history + 1))
        spatial_size: Board spatial dimension for the encoder
        policy_size: Policy output size for the encoder
        board_types: List of board type names this encoder supports
    """
    encoder_type: str
    encoder_class_name: str
    base_channels: int
    history_length: int
    in_channels: int
    spatial_size: int
    policy_size: int
    board_types: list[str]

    @property
    def frames(self) -> int:
        """Total number of frames (current + history)."""
        return self.history_length + 1


# Import policy sizes lazily to avoid circular imports
def _get_policy_sizes() -> dict[str, int]:
    """Get policy sizes from neural_net module."""
    from app.ai.neural_net import P_HEX, POLICY_SIZE_8x8, POLICY_SIZE_HEX8
    return {
        "P_HEX": P_HEX,
        "POLICY_SIZE_HEX8": POLICY_SIZE_HEX8,
        "POLICY_SIZE_8x8": POLICY_SIZE_8x8,
    }


# Registry of encoder configurations
# Key: (board_type_name, model_version)
# Value: EncoderConfig
_ENCODER_REGISTRY: dict[tuple[str, str], EncoderConfig] = {}


def _build_registry():
    """Build the encoder registry with all known configurations."""
    global _ENCODER_REGISTRY

    if _ENCODER_REGISTRY:
        return  # Already built

    sizes = _get_policy_sizes()

    # Hex V2 encoder (10 base channels)
    hex_v2_config = EncoderConfig(
        encoder_type="hex_v2",
        encoder_class_name="HexStateEncoder",
        base_channels=10,
        history_length=3,
        in_channels=40,  # 10 × 4
        spatial_size=25,
        policy_size=sizes["P_HEX"],
        board_types=["HEXAGONAL"],
    )

    hex8_v2_config = EncoderConfig(
        encoder_type="hex_v2",
        encoder_class_name="HexStateEncoder",
        base_channels=10,
        history_length=3,
        in_channels=40,  # 10 × 4
        spatial_size=9,
        policy_size=sizes["POLICY_SIZE_HEX8"],
        board_types=["HEX8"],
    )

    # Hex V3 encoder (16 base channels)
    hex_v3_config = EncoderConfig(
        encoder_type="hex_v3",
        encoder_class_name="HexStateEncoderV3",
        base_channels=16,
        history_length=3,
        in_channels=64,  # 16 × 4
        spatial_size=25,
        policy_size=sizes["P_HEX"],
        board_types=["HEXAGONAL"],
    )

    hex8_v3_config = EncoderConfig(
        encoder_type="hex_v3",
        encoder_class_name="HexStateEncoderV3",
        base_channels=16,
        history_length=3,
        in_channels=64,  # 16 × 4
        spatial_size=9,
        policy_size=sizes["POLICY_SIZE_HEX8"],
        board_types=["HEX8"],
    )

    # Square encoder (14 base channels)
    square8_config = EncoderConfig(
        encoder_type="square",
        encoder_class_name="NeuralNetAI",
        base_channels=14,
        history_length=3,
        in_channels=56,  # 14 × 4
        spatial_size=8,
        policy_size=sizes["POLICY_SIZE_8x8"],
        board_types=["SQUARE8"],
    )

    square19_config = EncoderConfig(
        encoder_type="square",
        encoder_class_name="NeuralNetAI",
        base_channels=14,
        history_length=3,
        in_channels=56,  # 14 × 4
        spatial_size=19,
        policy_size=0,  # Not typically used
        board_types=["SQUARE19"],
    )

    # Register all configurations
    _ENCODER_REGISTRY[("HEXAGONAL", "v2")] = hex_v2_config
    _ENCODER_REGISTRY[("HEXAGONAL", "v3")] = hex_v3_config
    _ENCODER_REGISTRY[("HEXAGONAL", "v4")] = hex_v3_config  # v4 uses same encoding as v3
    _ENCODER_REGISTRY[("HEX8", "v2")] = hex8_v2_config
    _ENCODER_REGISTRY[("HEX8", "v3")] = hex8_v3_config
    _ENCODER_REGISTRY[("HEX8", "v4")] = hex8_v3_config  # v4 uses same encoding as v3
    _ENCODER_REGISTRY[("SQUARE8", "v2")] = square8_config
    _ENCODER_REGISTRY[("SQUARE8", "v3")] = square8_config  # Same encoding for square
    _ENCODER_REGISTRY[("SQUARE19", "v2")] = square19_config
    _ENCODER_REGISTRY[("SQUARE19", "v3")] = square19_config


def get_encoder_config(
    board_type: "BoardType | str",
    model_version: str = "v2",
    history_length: int = 3,
) -> EncoderConfig:
    """Get encoder configuration for a board type and model version.

    Args:
        board_type: Board type (enum or string name)
        model_version: Model version ("v2", "v3", "v4")
        history_length: Number of history frames (adjusts in_channels if non-default)

    Returns:
        EncoderConfig with all encoder parameters

    Raises:
        ValueError: If no encoder config exists for the given combination
    """
    _build_registry()

    # Convert BoardType enum to string if needed
    if hasattr(board_type, "name"):
        board_type_name = board_type.name
    else:
        board_type_name = str(board_type).upper()

    key = (board_type_name, model_version)
    config = _ENCODER_REGISTRY.get(key)

    if config is None:
        available = sorted(set(k[0] for k in _ENCODER_REGISTRY.keys()))
        raise ValueError(
            f"No encoder configuration for board_type={board_type_name}, "
            f"model_version={model_version}.\n"
            f"Available board types: {available}\n"
            f"Available versions: v2, v3, v4"
        )

    # Adjust in_channels if history_length differs from default
    if history_length != config.history_length:
        frames = history_length + 1
        return EncoderConfig(
            encoder_type=config.encoder_type,
            encoder_class_name=config.encoder_class_name,
            base_channels=config.base_channels,
            history_length=history_length,
            in_channels=config.base_channels * frames,
            spatial_size=config.spatial_size,
            policy_size=config.policy_size,
            board_types=config.board_types,
        )

    return config


def get_expected_channels(
    board_type: "BoardType | str",
    model_version: str = "v2",
    history_length: int = 3,
) -> int:
    """Get expected input channel count for a board type and model version.

    This is a convenience function for quick channel validation.

    Args:
        board_type: Board type (enum or string name)
        model_version: Model version ("v2", "v3", "v4")
        history_length: Number of history frames

    Returns:
        Expected number of input channels
    """
    config = get_encoder_config(board_type, model_version, history_length)
    return config.in_channels


def validate_dataset_model_compatibility(
    dataset_in_channels: int,
    board_type: "BoardType | str",
    model_version: str = "v2",
    history_length: int = 3,
) -> None:
    """Validate that a dataset is compatible with a model configuration.

    Args:
        dataset_in_channels: Number of input channels in the dataset
        board_type: Board type for training
        model_version: Model version
        history_length: Number of history frames

    Raises:
        ValueError: If dataset channels don't match expected model channels
    """
    config = get_encoder_config(board_type, model_version, history_length)
    expected = config.in_channels

    if dataset_in_channels != expected:
        raise ValueError(
            f"Dataset/model channel mismatch:\n"
            f"  Dataset has {dataset_in_channels} channels\n"
            f"  Model {model_version} for {board_type} expects {expected} channels\n"
            f"  (Encoder: {config.encoder_type}, {config.base_channels} base × {config.frames} frames)\n"
            f"Solution: Use --model-version matching your data encoding, or\n"
            f"regenerate data with --encoder-version {model_version}"
        )


def list_encoder_configs() -> list[tuple[str, str, EncoderConfig]]:
    """List all registered encoder configurations.

    Returns:
        List of (board_type, model_version, config) tuples
    """
    _build_registry()
    return [(k[0], k[1], v) for k, v in sorted(_ENCODER_REGISTRY.items())]


# For backwards compatibility and convenience
ENCODER_CHANNEL_COUNTS = {
    "hex_v2": 40,
    "hex_v3": 64,
    "square": 56,
}

BASE_CHANNEL_COUNTS = {
    "hex_v2": 10,
    "hex_v3": 16,
    "square": 14,
}


def detect_model_version_from_channels(
    in_channels: int,
    board_type: str | None = None,
) -> str | None:
    """Detect model version from input channel count.

    This enables auto-detection of model version from NPZ training data,
    removing the need for manual --model-version flags.

    Channel to version mapping:
        - 40 channels: v2 (only hex v2 uses 40)
        - 64 channels: v3/v4 (hex v3/v4 uses 64)
        - 56 channels: v4 default (all square versions use 56)

    Args:
        in_channels: Number of input channels in the dataset
        board_type: Optional board type for disambiguation (e.g., "hex8", "square8")

    Returns:
        Model version string ("v2", "v4") if determinable, None if ambiguous

    Example:
        >>> detect_model_version_from_channels(40)
        'v2'
        >>> detect_model_version_from_channels(64)
        'v4'
        >>> detect_model_version_from_channels(56)
        'v4'
    """
    if in_channels == 40:
        # Only hex v2 uses 40 channels (10 base × 4 frames)
        return "v2"
    elif in_channels == 64:
        # Only hex v3/v4 uses 64 channels (16 base × 4 frames)
        return "v4"
    elif in_channels == 56:
        # All square versions use 56 channels (14 base × 4 frames)
        # Default to v4 as it's the latest
        return "v4"
    else:
        # Unknown channel count - could be custom history length
        return None
