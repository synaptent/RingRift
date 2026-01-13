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
        requires_heuristics: Whether this architecture requires heuristic features
        min_heuristic_features: Minimum number of heuristic features required (0=none, 21=fast, 49=full)
        encoder_version: Version string for encoder compatibility tracking
    """
    encoder_type: str
    encoder_class_name: str
    base_channels: int
    history_length: int
    in_channels: int
    spatial_size: int
    policy_size: int
    board_types: list[str]
    requires_heuristics: bool = False
    min_heuristic_features: int = 0
    encoder_version: str = "1.0"

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

    # Hex V5-Heavy encoder (16 base channels + 21 heuristic features required)
    hex_v5_heavy_config = EncoderConfig(
        encoder_type="hex_v5_heavy",
        encoder_class_name="HexNeuralNet_v5_Heavy",
        base_channels=16,
        history_length=3,
        in_channels=64,  # 16 × 4
        spatial_size=25,
        policy_size=sizes["P_HEX"],
        board_types=["HEXAGONAL"],
        requires_heuristics=True,
        min_heuristic_features=21,  # Fast heuristics minimum
        encoder_version="5.0",
    )

    hex8_v5_heavy_config = EncoderConfig(
        encoder_type="hex_v5_heavy",
        encoder_class_name="HexNeuralNet_v5_Heavy",
        base_channels=16,
        history_length=3,
        in_channels=64,  # 16 × 4
        spatial_size=9,
        policy_size=sizes["POLICY_SIZE_HEX8"],
        board_types=["HEX8"],
        requires_heuristics=True,
        min_heuristic_features=21,
        encoder_version="5.0",
    )

    # Hex V5 Heavy Large encoder (16 base channels + 49 full heuristic features required)
    # Note: "v6" is a deprecated alias for v5-heavy-large
    hex_v5_heavy_large_config = EncoderConfig(
        encoder_type="hex_v5_heavy_large",
        encoder_class_name="HexNeuralNet_v5_Heavy",  # Uses same architecture as V5 Heavy
        base_channels=16,
        history_length=3,
        in_channels=64,  # 16 × 4
        spatial_size=25,
        policy_size=sizes["P_HEX"],
        board_types=["HEXAGONAL"],
        requires_heuristics=True,
        min_heuristic_features=49,  # Full heuristics required
        encoder_version="5.2",
    )

    hex8_v5_heavy_large_config = EncoderConfig(
        encoder_type="hex_v5_heavy_large",
        encoder_class_name="HexNeuralNet_v5_Heavy",
        base_channels=16,
        history_length=3,
        in_channels=64,  # 16 × 4
        spatial_size=9,
        policy_size=sizes["POLICY_SIZE_HEX8"],
        board_types=["HEX8"],
        requires_heuristics=True,
        min_heuristic_features=49,
        encoder_version="5.2",
    )

    # Square V5-Heavy encoder (14 base channels + 21 heuristic features required)
    square8_v5_heavy_config = EncoderConfig(
        encoder_type="square_v5_heavy",
        encoder_class_name="RingRiftCNN_v5_Heavy",
        base_channels=14,
        history_length=3,
        in_channels=56,  # 14 × 4
        spatial_size=8,
        policy_size=sizes["POLICY_SIZE_8x8"],
        board_types=["SQUARE8"],
        requires_heuristics=True,
        min_heuristic_features=21,
        encoder_version="5.0",
    )

    # Register all configurations
    _ENCODER_REGISTRY[("HEXAGONAL", "v2")] = hex_v2_config
    _ENCODER_REGISTRY[("HEXAGONAL", "v3")] = hex_v3_config
    _ENCODER_REGISTRY[("HEXAGONAL", "v4")] = hex_v3_config  # v4 uses same encoding as v3
    _ENCODER_REGISTRY[("HEXAGONAL", "v5-heavy")] = hex_v5_heavy_config
    _ENCODER_REGISTRY[("HEXAGONAL", "v5-heavy-large")] = hex_v5_heavy_large_config
    _ENCODER_REGISTRY[("HEXAGONAL", "v5-heavy-xl")] = hex_v5_heavy_large_config  # Same encoder
    _ENCODER_REGISTRY[("HEXAGONAL", "v6")] = hex_v5_heavy_large_config  # Deprecated alias
    _ENCODER_REGISTRY[("HEXAGONAL", "v6-xl")] = hex_v5_heavy_large_config  # Deprecated alias
    _ENCODER_REGISTRY[("HEX8", "v2")] = hex8_v2_config
    _ENCODER_REGISTRY[("HEX8", "v3")] = hex8_v3_config
    _ENCODER_REGISTRY[("HEX8", "v4")] = hex8_v3_config  # v4 uses same encoding as v3
    _ENCODER_REGISTRY[("HEX8", "v5-heavy")] = hex8_v5_heavy_config
    _ENCODER_REGISTRY[("HEX8", "v5-heavy-large")] = hex8_v5_heavy_large_config
    _ENCODER_REGISTRY[("HEX8", "v5-heavy-xl")] = hex8_v5_heavy_large_config  # Same encoder
    _ENCODER_REGISTRY[("HEX8", "v6")] = hex8_v5_heavy_large_config  # Deprecated alias
    _ENCODER_REGISTRY[("HEX8", "v6-xl")] = hex8_v5_heavy_large_config  # Deprecated alias
    _ENCODER_REGISTRY[("SQUARE8", "v2")] = square8_config
    _ENCODER_REGISTRY[("SQUARE8", "v3")] = square8_config  # Same encoding for square
    _ENCODER_REGISTRY[("SQUARE8", "v5-heavy")] = square8_v5_heavy_config
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
        model_version: Model version ("v2", "v3", "v4", "v5-heavy", "v5-heavy-large", "v5-heavy-xl")
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
        versions = sorted(set(k[1] for k in _ENCODER_REGISTRY.keys()))
        raise ValueError(
            f"No encoder configuration for board_type={board_type_name}, "
            f"model_version={model_version}.\n"
            f"Available board types: {available}\n"
            f"Available versions: {versions}"
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


class EncoderVersionError(ValueError):
    """Raised when encoder version cannot be determined or is incompatible."""
    pass


def get_encoder_for_model(
    checkpoint_path: "str | Path",
    board_type: "BoardType | str | None" = None,
) -> Any:
    """Get the correct encoder for a model based on its stored metadata.

    This is the canonical way to get an encoder for inference. It reads
    the model's _versioning_metadata and returns a matching encoder instance.

    Reads _versioning_metadata.config from checkpoint to determine:
    - feature_version → encoder version (v2 or v3)
    - board_type → HexStateEncoder vs SquareStateEncoder
    - in_channels → validation that encoder matches

    Falls back to weight shape inference for models without metadata.

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        board_type: Override board type (uses metadata if None)

    Returns:
        Configured encoder instance (HexStateEncoder, HexStateEncoderV3, or SquareStateEncoder)

    Raises:
        EncoderVersionError: If encoder cannot be determined

    Example:
        >>> encoder = get_encoder_for_model('models/hex8_2p_gen3.pth')
        >>> features = encoder.encode_state(game_state)
    """
    import torch
    from pathlib import Path

    # Lazy imports to avoid circular dependencies
    from app.training.encoding import get_encoder_for_board_type
    from app.models import BoardType

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract metadata
    metadata = checkpoint.get("_versioning_metadata", {})
    config = metadata.get("config", {})

    # Determine board type
    resolved_board_type: BoardType
    if board_type is None:
        board_type_str = config.get("board_type") or metadata.get("board_type", "")
        if not board_type_str:
            # Try to infer from checkpoint path
            path_lower = str(checkpoint_path).lower()
            if "hex8" in path_lower:
                board_type_str = "hex8"
            elif "hexagonal" in path_lower:
                board_type_str = "hexagonal"
            elif "square8" in path_lower:
                board_type_str = "square8"
            elif "square19" in path_lower:
                board_type_str = "square19"
            else:
                board_type_str = "hex8"  # Default fallback
        resolved_board_type = BoardType(board_type_str)
    elif isinstance(board_type, str):
        resolved_board_type = BoardType(board_type)
    else:
        resolved_board_type = board_type

    # Determine encoder version from channel count
    in_channels = config.get("in_channels") or config.get("total_in_channels")
    feature_version = config.get("feature_version", 1)

    # Infer version from channels if available
    if in_channels:
        version = detect_model_version_from_channels(in_channels) or "v2"
    else:
        # Fallback: check conv1 weight shape
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if "conv1.weight" in state_dict:
            in_channels = state_dict["conv1.weight"].shape[1]
            version = detect_model_version_from_channels(in_channels) or "v2"
        else:
            version = "v2"  # Default

    encoder = get_encoder_for_board_type(resolved_board_type, version, feature_version)
    if encoder is None:
        raise EncoderVersionError(
            f"No encoder for board_type={resolved_board_type}, version={version}. "
            f"Model has in_channels={in_channels}, feature_version={feature_version}"
        )

    return encoder
