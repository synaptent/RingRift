"""Canonical registry mapping neural net architectures to their required encoders.

This module provides a single source of truth for the mapping between:
- Neural network architecture versions (v2, v3, v4, v5-heavy)
- Expected input channel counts (40, 64, 56)
- Corresponding encoder classes (HexStateEncoder, HexStateEncoderV3)

The registry enables automatic encoder selection based on loaded model weights,
ensuring MCTS and other inference code uses the correct encoder for any architecture.

Created: January 2026
Purpose: Fix encoder/model channel mismatches in MCTS evaluation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)


class ArchitectureVersion(Enum):
    """Supported neural network architecture versions."""

    V2 = "v2"                   # 40 channels (10 base × 4 frames)
    V2_LITE = "v2_lite"         # 40 channels (lighter model)
    V3 = "v3"                   # 64 channels (16 base × 4 frames)
    V3_LITE = "v3_lite"         # 64 channels (lighter model)
    V3_FLAT = "v3_flat"         # 64 channels (flat policy head)
    V4 = "v4"                   # 64 channels (attention-based)
    V5_HEAVY = "v5_heavy"       # 56 channels (14 base × 4 frames with heuristics)


@dataclass(frozen=True)
class ArchitectureSpec:
    """Specification for a neural network architecture."""

    version: ArchitectureVersion
    expected_channels: int
    encoder_name: str  # Class name of the encoder to use
    base_channels: int  # Channels before frame stacking
    frame_count: int   # Number of history frames (typically 4)
    description: str

    @property
    def class_names(self) -> Tuple[str, ...]:
        """Return the class names that match this architecture."""
        _version_to_classes = {
            ArchitectureVersion.V2: ("HexNeuralNet_v2", "RingRiftCNN_v2"),
            ArchitectureVersion.V2_LITE: ("HexNeuralNet_v2_Lite", "RingRiftCNN_v2_Lite"),
            ArchitectureVersion.V3: ("HexNeuralNet_v3", "RingRiftCNN_v3"),
            ArchitectureVersion.V3_LITE: ("HexNeuralNet_v3_Lite", "RingRiftCNN_v3_Lite"),
            ArchitectureVersion.V3_FLAT: ("HexNeuralNet_v3_Flat", "RingRiftCNN_v3_Flat"),
            ArchitectureVersion.V4: ("HexNeuralNet_v4", "RingRiftCNN_v4"),
            ArchitectureVersion.V5_HEAVY: ("HexNeuralNet_v5_Heavy", "RingRiftCNN_v5_Heavy"),
        }
        return _version_to_classes.get(self.version, ())


# Canonical registry: channel count -> architecture spec
ARCHITECTURE_REGISTRY: Dict[int, ArchitectureSpec] = {
    40: ArchitectureSpec(
        version=ArchitectureVersion.V2,
        expected_channels=40,
        encoder_name="HexStateEncoder",
        base_channels=10,
        frame_count=4,
        description="V2 standard (10 base × 4 frames)",
    ),
    64: ArchitectureSpec(
        version=ArchitectureVersion.V3,
        expected_channels=64,
        encoder_name="HexStateEncoderV3",
        base_channels=16,
        frame_count=4,
        description="V3/V4 enhanced (16 base × 4 frames)",
    ),
    56: ArchitectureSpec(
        version=ArchitectureVersion.V5_HEAVY,
        expected_channels=56,
        encoder_name="HexStateEncoderV5",
        base_channels=14,
        frame_count=4,
        description="V5-heavy with heuristics (14 base × 4 frames)",
    ),
    # Lite variants with 36 channels (12 base × 3 frames)
    36: ArchitectureSpec(
        version=ArchitectureVersion.V2_LITE,
        expected_channels=36,
        encoder_name="HexStateEncoder",
        base_channels=12,
        frame_count=3,
        description="V2-lite (12 base × 3 frames)",
    ),
    # V3-lite with 44 channels (12 base × 3 frames + 8 phase/chain)
    44: ArchitectureSpec(
        version=ArchitectureVersion.V3_LITE,
        expected_channels=44,
        encoder_name="HexStateEncoderV3Lite",
        base_channels=12,
        frame_count=3,
        description="V3-lite (12 base × 3 frames + 8 extras)",
    ),
}


def get_expected_channels_from_model(model: "nn.Module") -> Optional[int]:
    """
    Detect the expected input channels from a loaded model.

    This examines the model's conv1/initial_conv weight shape to determine
    how many input channels the model expects.

    Args:
        model: A loaded PyTorch neural network model

    Returns:
        The number of input channels expected, or None if not detectable
    """
    try:
        # Check common layer names for the first convolution
        for name, param in model.named_parameters():
            if any(key in name for key in ('conv1.weight', 'initial_conv.weight')):
                # Conv weight shape is [out_channels, in_channels, H, W]
                in_channels = param.shape[1]
                logger.debug(f"Detected {in_channels} channels from {name}")
                return int(in_channels)

        # Try accessing the in_channels attribute directly
        if hasattr(model, 'in_channels'):
            return int(model.in_channels)

        return None
    except Exception as e:
        logger.warning(f"Failed to detect model channels: {e}")
        return None


def get_architecture_spec(channels: int) -> Optional[ArchitectureSpec]:
    """
    Get the architecture specification for a given channel count.

    Args:
        channels: Number of input channels

    Returns:
        ArchitectureSpec if found, None otherwise
    """
    return ARCHITECTURE_REGISTRY.get(channels)


def get_encoder_class_for_channels(channels: int) -> Optional[Type[Any]]:
    """
    Get the encoder class for a given channel count.

    Args:
        channels: Number of input channels

    Returns:
        The encoder class, or None if not found
    """
    spec = get_architecture_spec(channels)
    if spec is None:
        logger.warning(f"No architecture spec for {channels} channels")
        return None

    # Import the encoder class dynamically
    try:
        if spec.encoder_name == "HexStateEncoder":
            from app.training.encoding import HexStateEncoder
            return HexStateEncoder
        elif spec.encoder_name == "HexStateEncoderV3":
            from app.training.encoding import HexStateEncoderV3
            return HexStateEncoderV3
        elif spec.encoder_name == "HexStateEncoderV5":
            # V5 may use a specialized encoder or fall back to V3
            try:
                from app.training.encoding import HexStateEncoderV5
                return HexStateEncoderV5
            except ImportError:
                # Fall back to V3 encoder if V5 not available
                from app.training.encoding import HexStateEncoderV3
                logger.debug("HexStateEncoderV5 not found, using V3")
                return HexStateEncoderV3
        elif spec.encoder_name == "HexStateEncoderV3Lite":
            # Lite variant may fall back to base encoder
            try:
                from app.training.encoding import HexStateEncoderV3Lite
                return HexStateEncoderV3Lite
            except ImportError:
                from app.training.encoding import HexStateEncoder
                return HexStateEncoder
        else:
            logger.warning(f"Unknown encoder name: {spec.encoder_name}")
            return None
    except ImportError as e:
        logger.error(f"Failed to import encoder {spec.encoder_name}: {e}")
        return None


def get_encoder_for_model(model: "nn.Module") -> Optional[Any]:
    """
    Get an instantiated encoder appropriate for a loaded model.

    This is a convenience function that detects the model's channel count
    and returns an appropriate encoder instance.

    Args:
        model: A loaded PyTorch neural network model

    Returns:
        An encoder instance, or None if not determinable
    """
    channels = get_expected_channels_from_model(model)
    if channels is None:
        logger.warning("Could not detect model channels, using default encoder")
        return None

    encoder_class = get_encoder_class_for_channels(channels)
    if encoder_class is None:
        return None

    try:
        # Instantiate with default parameters (board_size and num_players
        # will be set when encode() is called)
        return encoder_class()
    except Exception as e:
        logger.error(f"Failed to instantiate encoder: {e}")
        return None


def validate_encoder_model_match(
    encoder: Any,
    model: "nn.Module",
) -> Tuple[bool, str]:
    """
    Validate that an encoder produces the correct number of channels for a model.

    Args:
        encoder: An encoder instance
        model: A loaded neural network model

    Returns:
        Tuple of (is_valid, error_message)
    """
    expected = get_expected_channels_from_model(model)
    if expected is None:
        return False, "Could not detect model's expected channels"

    # Check encoder's output channels
    # Most encoders have NUM_CHANNELS or num_channels attribute
    actual = None
    if hasattr(encoder, 'NUM_CHANNELS'):
        base = encoder.NUM_CHANNELS
        frames = getattr(encoder, 'history_length', 3) + 1
        actual = base * frames
    elif hasattr(encoder, 'num_channels'):
        actual = encoder.num_channels

    if actual is None:
        return False, "Could not detect encoder's output channels"

    if actual != expected:
        return False, f"Encoder produces {actual} channels but model expects {expected}"

    return True, "OK"


def get_architecture_from_class_name(class_name: str) -> Optional[ArchitectureSpec]:
    """
    Get architecture spec from a model class name.

    Args:
        class_name: The model class name (e.g., "HexNeuralNet_v2")

    Returns:
        ArchitectureSpec if found, None otherwise
    """
    for spec in ARCHITECTURE_REGISTRY.values():
        if class_name in spec.class_names:
            return spec
    return None


# Convenience mappings for common use cases
CHANNEL_TO_ENCODER_NAME: Dict[int, str] = {
    spec.expected_channels: spec.encoder_name
    for spec in ARCHITECTURE_REGISTRY.values()
}

ENCODER_NAME_TO_CHANNELS: Dict[str, int] = {
    spec.encoder_name: spec.expected_channels
    for spec in ARCHITECTURE_REGISTRY.values()
}
