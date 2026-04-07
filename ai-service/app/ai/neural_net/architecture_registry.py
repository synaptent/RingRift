"""Canonical registry mapping neural net architectures to their required encoders.

This module provides a single source of truth for the mapping between:
- Neural network architecture versions (v2, v3, v4, v5-heavy)
- Expected input channel counts for unambiguous channel families
- Corresponding encoder classes

There are two lookup modes:
- channel-family lookup via ``ARCHITECTURE_REGISTRY`` for unambiguous cases
  (40ch hex v2, 64ch hex v3/v4 family, 56ch square family),
- class-name lookup via ``get_architecture_from_class_name()`` for ambiguous
  cases such as hex ``v5-heavy`` vs square ``56ch`` families.

This keeps MCTS/inference encoder selection truthful instead of forcing one
fake interpretation onto every 56- or 64-channel model.

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
    V5_HEAVY = "v5_heavy"       # Hex: 64 channels, Square: 56 channels
    SQUARE_FAMILY = "square_family"  # Generic square encoder family (56 channels)


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
            ArchitectureVersion.V2: ("HexNeuralNet_v2",),
            ArchitectureVersion.V2_LITE: ("HexNeuralNet_v2_Lite",),
            ArchitectureVersion.V3: ("HexNeuralNet_v3",),
            ArchitectureVersion.V3_LITE: ("HexNeuralNet_v3_Lite",),
            ArchitectureVersion.V3_FLAT: ("HexNeuralNet_v3_Flat",),
            ArchitectureVersion.V4: ("HexNeuralNet_v4",),
            ArchitectureVersion.V5_HEAVY: ("HexNeuralNet_v5_Heavy",),
            ArchitectureVersion.SQUARE_FAMILY: (),
        }
        return _version_to_classes.get(self.version, ())


HEX_V2_SPEC = ArchitectureSpec(
    version=ArchitectureVersion.V2,
    expected_channels=40,
    encoder_name="HexStateEncoder",
    base_channels=10,
    frame_count=4,
    description="Hex V2 standard (10 base × 4 frames)",
)

HEX_V3_FAMILY_SPEC = ArchitectureSpec(
    version=ArchitectureVersion.V3,
    expected_channels=64,
    encoder_name="HexStateEncoderV3",
    base_channels=16,
    frame_count=4,
    description="Hex V3/V4/V5-heavy family (16 base × 4 frames)",
)

SQUARE_FAMILY_SPEC = ArchitectureSpec(
    version=ArchitectureVersion.SQUARE_FAMILY,
    expected_channels=56,
    encoder_name="SquareStateEncoder",
    base_channels=14,
    frame_count=4,
    description="Square encoder family (14 base × 4 frames)",
)

V2_LITE_SPEC = ArchitectureSpec(
    version=ArchitectureVersion.V2_LITE,
    expected_channels=36,
    encoder_name="HexStateEncoder",
    base_channels=12,
    frame_count=3,
    description="V2-lite (12 base × 3 frames)",
)

V3_LITE_SPEC = ArchitectureSpec(
    version=ArchitectureVersion.V3_LITE,
    expected_channels=44,
    encoder_name="HexStateEncoderV3Lite",
    base_channels=12,
    frame_count=3,
    description="V3-lite (12 base × 3 frames + 8 extras)",
)

_CLASS_NAME_TO_SPEC: Dict[str, ArchitectureSpec] = {
    "HexNeuralNet_v2": HEX_V2_SPEC,
    "HexNeuralNet_v2_Lite": V2_LITE_SPEC,
    "HexNeuralNet_v3": ArchitectureSpec(
        version=ArchitectureVersion.V3,
        expected_channels=64,
        encoder_name="HexStateEncoderV3",
        base_channels=16,
        frame_count=4,
        description="Hex V3 (16 base × 4 frames)",
    ),
    "HexNeuralNet_v3_Lite": V3_LITE_SPEC,
    "HexNeuralNet_v3_Flat": ArchitectureSpec(
        version=ArchitectureVersion.V3_FLAT,
        expected_channels=64,
        encoder_name="HexStateEncoderV3",
        base_channels=16,
        frame_count=4,
        description="Hex V3 flat policy (16 base × 4 frames)",
    ),
    "HexNeuralNet_v4": ArchitectureSpec(
        version=ArchitectureVersion.V4,
        expected_channels=64,
        encoder_name="HexStateEncoderV3",
        base_channels=16,
        frame_count=4,
        description="Hex V4 attention family (16 base × 4 frames)",
    ),
    "HexNeuralNet_v5_Heavy": ArchitectureSpec(
        version=ArchitectureVersion.V5_HEAVY,
        expected_channels=64,
        encoder_name="HexStateEncoderV3",
        base_channels=16,
        frame_count=4,
        description="Hex V5-heavy family (16 base × 4 frames + heuristics)",
    ),
    "RingRiftCNN_v2": ArchitectureSpec(
        version=ArchitectureVersion.V2,
        expected_channels=56,
        encoder_name="SquareStateEncoder",
        base_channels=14,
        frame_count=4,
        description="Square V2 family (14 base × 4 frames)",
    ),
    "RingRiftCNN_v2_Lite": ArchitectureSpec(
        version=ArchitectureVersion.V2_LITE,
        expected_channels=56,
        encoder_name="SquareStateEncoder",
        base_channels=14,
        frame_count=4,
        description="Square V2-lite family (14 base × 4 frames)",
    ),
    "RingRiftCNN_v3": ArchitectureSpec(
        version=ArchitectureVersion.V3,
        expected_channels=56,
        encoder_name="SquareStateEncoder",
        base_channels=14,
        frame_count=4,
        description="Square V3 family (14 base × 4 frames)",
    ),
    "RingRiftCNN_v3_Lite": ArchitectureSpec(
        version=ArchitectureVersion.V3_LITE,
        expected_channels=56,
        encoder_name="SquareStateEncoder",
        base_channels=14,
        frame_count=4,
        description="Square V3-lite family (14 base × 4 frames)",
    ),
    "RingRiftCNN_v3_Flat": ArchitectureSpec(
        version=ArchitectureVersion.V3_FLAT,
        expected_channels=56,
        encoder_name="SquareStateEncoder",
        base_channels=14,
        frame_count=4,
        description="Square V3 flat policy family (14 base × 4 frames)",
    ),
    "RingRiftCNN_v4": ArchitectureSpec(
        version=ArchitectureVersion.V4,
        expected_channels=56,
        encoder_name="SquareStateEncoder",
        base_channels=14,
        frame_count=4,
        description="Square V4 family (14 base × 4 frames)",
    ),
    "RingRiftCNN_v5_Heavy": ArchitectureSpec(
        version=ArchitectureVersion.V5_HEAVY,
        expected_channels=56,
        encoder_name="SquareStateEncoder",
        base_channels=14,
        frame_count=4,
        description="Square V5-heavy family (14 base × 4 frames + heuristics)",
    ),
}


# Canonical registry: channel count -> generic architecture spec
ARCHITECTURE_REGISTRY: Dict[int, ArchitectureSpec] = {
    40: HEX_V2_SPEC,
    64: HEX_V3_FAMILY_SPEC,
    56: SQUARE_FAMILY_SPEC,
    36: V2_LITE_SPEC,
    44: V3_LITE_SPEC,
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
        elif spec.encoder_name == "SquareStateEncoder":
            from app.training.encoding import SquareStateEncoder
            return SquareStateEncoder
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
    class_name = model.__class__.__name__
    spec = _CLASS_NAME_TO_SPEC.get(class_name)

    if spec is None:
        channels = get_expected_channels_from_model(model)
        if channels is None:
            logger.warning("Could not detect model channels, using default encoder")
            return None
        spec = get_architecture_spec(channels)
        if spec is None:
            return None

    encoder_class = get_encoder_class_for_channels(spec.expected_channels)
    if encoder_class is None:
        return None

    try:
        if spec.encoder_name == "SquareStateEncoder":
            from app.models import BoardType

            board_size = int(getattr(model, "board_size", 8))
            board_type = BoardType.SQUARE19 if board_size == 19 else BoardType.SQUARE8
            return encoder_class(
                board_type=board_type,
                board_size=board_size,
                feature_version=2,
            )
        if spec.encoder_name in {
            "HexStateEncoder",
            "HexStateEncoderV3",
            "HexStateEncoderV3Lite",
        }:
            from app.ai.neural_net import POLICY_SIZE_HEX8, P_HEX

            board_size = int(getattr(model, "board_size", 25))
            policy_size = POLICY_SIZE_HEX8 if board_size == 9 else P_HEX
            kwargs: dict[str, Any] = {
                "board_size": board_size,
                "policy_size": policy_size,
            }
            if spec.expected_channels == 64:
                kwargs["feature_version"] = 2
            return encoder_class(**kwargs)
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


def get_validated_encoder(
    model: "nn.Module",
    *,
    board_type: Optional[str] = None,
    board_size: Optional[int] = None,
    policy_size: Optional[int] = None,
) -> Any:
    """Unified encoder selection with mandatory channel validation.

    Mar 2026: Single entry point for encoder selection that replaces scattered
    instantiation paths. Detects model channels, instantiates the correct
    encoder, and validates the match — preventing the silent encoding mismatches
    that caused models to train on malformed input.

    Args:
        model: A loaded PyTorch neural network model
        board_type: Optional board type for board-size-aware encoders
        board_size: Optional board size override
        policy_size: Optional policy size override

    Returns:
        An encoder instance guaranteed to match the model's channel count.

    Raises:
        ValueError: If encoder/model channels don't match or can't be determined.
    """
    channels = get_expected_channels_from_model(model)
    if channels is None:
        raise ValueError(
            "Cannot detect model input channels from conv1.weight. "
            "Model may have non-standard architecture."
        )

    encoder = get_encoder_for_model(model)
    if encoder is None:
        raise ValueError(
            f"No encoder found for {channels}-channel model. "
            f"Supported: {list(CHANNEL_TO_ENCODER_NAME.keys())}"
        )

    # Apply board-specific parameters if provided
    if board_size is not None and hasattr(encoder, 'board_size'):
        encoder.board_size = board_size
    if policy_size is not None and hasattr(encoder, 'policy_size'):
        encoder.policy_size = policy_size

    # Mandatory validation — raise if channels don't match
    is_valid, error = validate_encoder_model_match(encoder, model)
    if not is_valid:
        raise ValueError(
            f"Encoder/model channel mismatch: {error}. "
            f"Model expects {channels} channels. "
            "This prevents training on malformed input."
        )

    logger.info(
        f"[EncoderContract] Validated encoder for {channels}ch model "
        f"({encoder.__class__.__name__})"
    )
    return encoder


def get_architecture_from_class_name(class_name: str) -> Optional[ArchitectureSpec]:
    """
    Get architecture spec from a model class name.

    Args:
        class_name: The model class name (e.g., "HexNeuralNet_v2")

    Returns:
        ArchitectureSpec if found, None otherwise
    """
    return _CLASS_NAME_TO_SPEC.get(class_name)


# Convenience mappings for common use cases
CHANNEL_TO_ENCODER_NAME: Dict[int, str] = {
    spec.expected_channels: spec.encoder_name
    for spec in ARCHITECTURE_REGISTRY.values()
}

ENCODER_NAME_TO_CHANNELS: Dict[str, int] = {
    spec.encoder_name: spec.expected_channels
    for spec in ARCHITECTURE_REGISTRY.values()
}


def get_encoder_version_from_checkpoint(checkpoint_path: str) -> Optional[str]:
    """
    Detect the encoder version required by a model checkpoint.

    This reads the model checkpoint and determines which encoder version
    (v2 or v3) is required based on the model's input channel count.

    Args:
        checkpoint_path: Path to a model .pth file

    Returns:
        "v2" for 40-channel models (HexNeuralNet_v2)
        "v3" for 64-channel models (HexNeuralNet_v3/v4)
        None if detection fails

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    import torch
    from pathlib import Path

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        # Load checkpoint (CPU only, for metadata inspection)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # April 2026: Prefer explicit versioning metadata over channel-count inference.
        # Exact architecture is ambiguous for some channel families (for example
        # 64ch hex v3/v4/v5-heavy), so use metadata when present.
        if isinstance(checkpoint, dict):
            meta = checkpoint.get("_versioning_metadata", {})
            arch_ver = meta.get("architecture_version", "")
            if arch_ver.startswith("v2"):
                return "v2"
            elif arch_ver.startswith("v3"):
                return "v3"
            elif arch_ver.startswith("v4"):
                return "v3"  # v4 uses v3 encoder
            elif "v5" in arch_ver:
                return "v5-heavy"

        # Fallback: infer from conv1 weight shape
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        else:
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else None

        if state_dict is None:
            logger.warning(f"Could not extract state_dict from {checkpoint_path}")
            return None

        # Find conv1 weight to detect input channels
        for key in state_dict:
            if "conv1.weight" in key or "initial_conv.weight" in key:
                weight = state_dict[key]
                in_channels = weight.shape[1]
                logger.debug(f"Detected {in_channels} channels from {key}")

                # Map channels to encoder family. Square 56ch models use the
                # square encoder family; hex 64ch models use the V3-family
                # encoder path (covering v3/v4/v5-heavy).
                if in_channels == 40:
                    return "v2"
                elif in_channels == 64:
                    return "v3"
                elif in_channels == 56:
                    return "v2"
                else:
                    logger.warning(f"Unknown channel count {in_channels}, defaulting to v3")
                    return "v3"

        logger.warning(f"Could not find conv1 layer in {checkpoint_path}")
        return None

    except Exception as e:
        logger.error(f"Failed to detect encoder version from {checkpoint_path}: {e}")
        return None


def get_model_version_from_checkpoint(checkpoint_path: str) -> Optional[str]:
    """
    Detect the model architecture version from a checkpoint.

    This examines the model structure to determine which architecture class
    was used (v2, v3, v4, v5-heavy, etc.) based on:
    - Number of value FC layers (2 for v2/v3, 3 for v4)
    - Number of residual blocks
    - Presence of attention layers
    - Presence of SE blocks

    Args:
        checkpoint_path: Path to a model .pth file

    Returns:
        Model version string ("v2", "v3", "v4", "v5-heavy") or None
    """
    import torch
    from pathlib import Path

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # April 2026: Prefer explicit versioning metadata over weight-shape inference.
        if isinstance(checkpoint, dict):
            meta = checkpoint.get("_versioning_metadata", {})
            arch_ver = meta.get("architecture_version", "")
            if arch_ver.startswith("v2"):
                return "v2"
            elif arch_ver.startswith("v3"):
                return "v3"
            elif arch_ver.startswith("v4"):
                return "v4"
            elif "v5" in arch_ver:
                return "v5-heavy"

        # Fallback: infer from model weight structure
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        else:
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else None

        if state_dict is None:
            return None

        keys = list(state_dict.keys())

        # Count architecture indicators
        value_fc_layers = [k for k in keys if "value_fc" in k and "weight" in k]
        has_value_fc3 = any("value_fc3" in k for k in keys)
        has_heuristic_encoder = any("heuristic_encoder" in k for k in keys)

        if has_heuristic_encoder:
            return "v5-heavy"
        elif has_value_fc3:
            return "v4"
        elif len(value_fc_layers) == 2:
            # Check input channels to distinguish square-family from hex-family
            # checkpoints when structural signals are otherwise limited.
            for key in keys:
                if "conv1.weight" in key:
                    weight = state_dict[key]
                    in_channels = weight.shape[1]
                    if in_channels == 40:
                        return "v2"
                    elif in_channels == 64:
                        return "v3"
                    elif in_channels == 56:
                        return "v2"
                    break
            return "v2"
        else:
            return "v2"

    except Exception as e:
        logger.error(f"Failed to detect model version from {checkpoint_path}: {e}")
        return None


def validate_export_architecture_match(
    canonical_model_path: str,
    encoder_version: str,
) -> Tuple[bool, str]:
    """
    Validate that an export encoder version matches the canonical model.

    This is a fail-fast check to prevent architecture mismatches during export.

    Args:
        canonical_model_path: Path to the canonical model checkpoint
        encoder_version: The encoder version being used for export ("v2" or "v3")

    Returns:
        Tuple of (is_valid, error_message)
    """
    expected = get_encoder_version_from_checkpoint(canonical_model_path)
    if expected is None:
        return False, f"Could not detect architecture from {canonical_model_path}"

    if expected != encoder_version:
        return False, (
            f"Architecture mismatch: canonical model {canonical_model_path} uses "
            f"encoder {expected}, but export is configured for {encoder_version}. "
            f"Use --encoder-version {expected} or let --canonical-model auto-detect."
        )

    return True, "OK"
