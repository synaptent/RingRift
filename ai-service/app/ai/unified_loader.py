"""Unified Model Loader for RingRift Neural Networks.

This module provides automatic architecture detection and loading for ALL
RingRift model types (NNUE, CNN v2/v3/v4, Hex v2/v3, and experimental).

The key insight is that model architectures can be distinguished by their
state_dict key patterns:
- NNUE: accumulator.*, hidden_blocks.*
- NNUE+Policy: accumulator.*, from_head.*, to_head.*
- CNN v2/v3/v4: conv1.*, res_blocks.*, policy_fc.*
- Hex: conv1.*, hex_mask, res_blocks.*

Usage:
    loader = UnifiedModelLoader()
    loaded = loader.load("models/my_model.pth", BoardType.SQUARE8)
    # loaded.model is ready to use
    # loaded.architecture tells you which type
"""

from __future__ import annotations

import gc
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from app.models import BoardType

logger = logging.getLogger(__name__)


class ModelArchitecture(Enum):
    """Detected model architecture types."""

    NNUE_VALUE_ONLY = auto()  # RingRiftNNUE - value head only
    NNUE_WITH_POLICY = auto()  # RingRiftNNUEWithPolicy - value + from/to policy
    CNN_V2 = auto()  # RingRiftCNN_v2 - SE blocks, FC policy
    CNN_V2_LITE = auto()  # RingRiftCNN_v2_Lite - memory-efficient
    CNN_V3 = auto()  # RingRiftCNN_v3 - spatial policy heads
    CNN_V3_LITE = auto()  # RingRiftCNN_v3_Lite
    CNN_V4 = auto()  # RingRiftCNN_v4 - NAS with attention
    HEX_V2 = auto()  # HexNeuralNet_v2 - hex-specific SE
    HEX_V2_LITE = auto()  # HexNeuralNet_v2_Lite
    HEX_V3 = auto()  # HexNeuralNet_v3 - hex spatial policy
    HEX_V3_LITE = auto()  # HexNeuralNet_v3_Lite
    UNKNOWN = auto()  # Fallback for future/experimental architectures


# Board type to policy size mapping (board-aware encoding)
# These MUST match app.ai.neural_net.constants.get_policy_size_for_board()
POLICY_SIZE_MAP = {
    BoardType.SQUARE8: 7000,     # Was 4672 (legacy)
    BoardType.SQUARE19: 67000,   # Was 43681 (legacy)
    BoardType.HEX8: 4500,        # Was 13953 (legacy)
    BoardType.HEXAGONAL: 91876,  # Was 55809 (legacy)
}

# Policy size to board type reverse mapping
POLICY_SIZE_TO_BOARD = {v: k for k, v in POLICY_SIZE_MAP.items()}


@dataclass
class ArchitectureSignature:
    """Signature patterns for architecture detection."""

    required_prefixes: list[str]  # Keys that MUST be present
    forbidden_prefixes: list[str]  # Keys that MUST NOT be present
    architecture: ModelArchitecture
    priority: int = 0  # Higher = check first (for disambiguation)


# Architecture detection signatures ordered by priority
ARCHITECTURE_SIGNATURES = [
    # NNUE+Policy: has both accumulator AND from_head/to_head
    ArchitectureSignature(
        required_prefixes=["accumulator.", "from_head."],
        forbidden_prefixes=["conv1."],
        architecture=ModelArchitecture.NNUE_WITH_POLICY,
        priority=100,
    ),
    # NNUE value-only: has accumulator but NO from_head/to_head/conv
    ArchitectureSignature(
        required_prefixes=["accumulator."],
        forbidden_prefixes=["from_head.", "to_head.", "conv1."],
        architecture=ModelArchitecture.NNUE_VALUE_ONLY,
        priority=90,
    ),
    # Hex v3: has hex_mask and typically spatial policy
    ArchitectureSignature(
        required_prefixes=["conv1.", "res_blocks."],
        forbidden_prefixes=[],
        architecture=ModelArchitecture.HEX_V2,  # Default hex, refined later
        priority=70,
    ),
    # CNN v4: has attention blocks
    ArchitectureSignature(
        required_prefixes=["conv1.", "res_blocks."],
        forbidden_prefixes=["hex_mask"],
        architecture=ModelArchitecture.CNN_V2,  # Default CNN, refined later
        priority=60,
    ),
]


@dataclass
class InferredModelConfig:
    """Configuration inferred from checkpoint weights."""

    architecture: ModelArchitecture
    board_type: BoardType | None = None
    num_players: int = 2
    num_res_blocks: int = 12
    num_filters: int = 192
    policy_size: int = 4672
    is_lite_variant: bool = False
    hidden_dim: int = 256
    input_channels: int = 14  # BASE input channels (not total)
    global_features: int = 20
    hex_radius: int = 4
    policy_intermediate: int = 384  # Inferred from policy_fc1
    value_intermediate: int = 128  # Inferred from value_fc1
    history_length: int = 3  # Inferred from total_in_channels / base_channels


@dataclass
class LoadedModel:
    """Result of loading a model through UnifiedModelLoader."""

    model: nn.Module
    architecture: ModelArchitecture
    config: InferredModelConfig
    checkpoint_path: Path
    device: torch.device
    has_policy: bool  # Whether model outputs policy
    has_value: bool = True  # Whether model outputs value (always True)
    is_multiplayer_value: bool = False  # Whether value is per-player


def detect_architecture(state_dict: dict[str, Any]) -> ModelArchitecture:
    """Detect model architecture from checkpoint state dict keys.

    Examines the keys in a PyTorch state_dict to determine which
    architecture family the model belongs to.

    Args:
        state_dict: The model's state dictionary (key -> tensor mapping)

    Returns:
        The detected ModelArchitecture enum value
    """
    keys = set(state_dict.keys())

    def has_prefix(prefix: str) -> bool:
        return any(k.startswith(prefix) for k in keys)

    # Sort by priority (highest first) for correct disambiguation
    sorted_sigs = sorted(ARCHITECTURE_SIGNATURES, key=lambda s: -s.priority)

    for sig in sorted_sigs:
        # Check required prefixes
        has_all_required = all(has_prefix(p) for p in sig.required_prefixes)
        # Check forbidden prefixes
        has_forbidden = any(has_prefix(p) for p in sig.forbidden_prefixes)

        if has_all_required and not has_forbidden:
            arch = sig.architecture

            # Refine hex vs square detection
            if arch in (ModelArchitecture.CNN_V2, ModelArchitecture.HEX_V2):
                if has_prefix("hex_mask"):
                    # Detect V3 vs V2 hex: V3 has spatial policy heads (placement_conv, movement_conv)
                    if has_prefix("placement_conv.") and has_prefix("movement_conv."):
                        arch = ModelArchitecture.HEX_V3
                    else:
                        arch = ModelArchitecture.HEX_V2
                else:
                    arch = ModelArchitecture.CNN_V2

            # Refine CNN_V2 -> CNN_V3/V4 detection
            # V3 models have spatial policy heads (placement_conv, movement_conv)
            # V4 models have attention blocks (keys containing "attn" or "attention")
            if arch == ModelArchitecture.CNN_V2:
                if has_prefix("placement_conv.") and has_prefix("movement_conv."):
                    arch = ModelArchitecture.CNN_V3
                elif any("attn" in k or "attention" in k for k in keys):
                    arch = ModelArchitecture.CNN_V4

            return arch

    return ModelArchitecture.UNKNOWN


def infer_config_from_checkpoint(
    state_dict: dict[str, Any],
    architecture: ModelArchitecture,
    metadata: dict[str, Any] | None = None,
) -> InferredModelConfig:
    """Infer complete model configuration from checkpoint.

    This enables loading ANY checkpoint without knowing its original config.
    """
    config = InferredModelConfig(architecture=architecture)

    metadata = metadata or {}

    # Extract metadata hints
    if "config" in metadata:
        meta_config = metadata["config"]
        if "board_size" in meta_config:
            size = meta_config["board_size"]
            if size == 8:
                config.board_type = BoardType.SQUARE8
            elif size == 19:
                config.board_type = BoardType.SQUARE19
        if "num_players" in meta_config:
            config.num_players = meta_config["num_players"]

    # Infer from conv1 shape: [out_channels, total_in_channels, H, W]
    if "conv1.weight" in state_dict:
        conv1_shape = state_dict["conv1.weight"].shape
        config.num_filters = conv1_shape[0]
        total_in_channels = conv1_shape[1]

        # Infer base input_channels and history_length from total
        # Common combinations: 14*(3+1)=56, 12*(2+1)=36, 14*(2+1)=42
        for hist_len in [3, 2, 4, 1]:
            divisor = hist_len + 1
            if total_in_channels % divisor == 0:
                base_channels = total_in_channels // divisor
                if base_channels in [12, 14, 16]:  # Common base channel counts
                    config.input_channels = base_channels
                    config.history_length = hist_len
                    break
        else:
            # Fallback: assume history_length=3, use total as-is
            config.input_channels = total_in_channels
            config.history_length = 0  # Will use model's default

        # Lite variant detection: fewer filters AND smaller intermediate layers
        # Don't just rely on filter count alone

    # Count residual blocks
    max_block_idx = -1
    for key in state_dict:
        if key.startswith("res_blocks.") and ".conv1.weight" in key:
            try:
                idx = int(key.split(".")[1])
                max_block_idx = max(max_block_idx, idx)
            except (ValueError, IndexError):
                pass
    if max_block_idx >= 0:
        config.num_res_blocks = max_block_idx + 1

    # Infer policy size from policy head
    for key in ["policy_fc2.weight", "policy_fc.weight", "policy_fc1.weight"]:
        if key in state_dict:
            config.policy_size = state_dict[key].shape[0]
            break

    # Infer board type from policy size
    if config.policy_size in POLICY_SIZE_TO_BOARD:
        config.board_type = POLICY_SIZE_TO_BOARD[config.policy_size]

    # Detect hex from hex_mask
    if "hex_mask" in state_dict:
        hex_mask_shape = state_dict["hex_mask"].shape
        # hex_mask is [1, 1, size, size]
        if len(hex_mask_shape) >= 3:
            hex_size = hex_mask_shape[-1]
            if hex_size == 9:
                config.board_type = BoardType.HEX8
                config.hex_radius = 4
            elif hex_size == 25:
                config.board_type = BoardType.HEXAGONAL
                config.hex_radius = 12

    # Infer num_players from value head output
    # V4 has 3-layer value head: fc1 -> fc2 -> fc3 (output)
    # V2/V3 has 2-layer value head: fc1 -> fc2 (output)
    if "value_fc3.weight" in state_dict:
        # V4 model: fc3 outputs to num_players
        config.num_players = state_dict["value_fc3.weight"].shape[0]
    elif "value_fc2.weight" in state_dict:
        # V2/V3 model: fc2 outputs to num_players
        config.num_players = state_dict["value_fc2.weight"].shape[0]

    # NNUE-specific: hidden_dim from accumulator
    if "accumulator.weight" in state_dict:
        config.hidden_dim = state_dict["accumulator.weight"].shape[0]

    # Infer intermediate layer sizes from FC layers
    if "policy_fc1.weight" in state_dict:
        config.policy_intermediate = state_dict["policy_fc1.weight"].shape[0]
    if "value_fc1.weight" in state_dict:
        config.value_intermediate = state_dict["value_fc1.weight"].shape[0]

    # Lite variant detection: based on BOTH filter count AND intermediate sizes
    # Lite variants have: 96 filters, 192 policy_intermediate, 64 value_intermediate
    # Full variants have: 192 filters, 384 policy_intermediate, 128 value_intermediate
    config.is_lite_variant = (
        config.num_filters <= 96
        and config.policy_intermediate <= 200
        and config.value_intermediate <= 80
    )

    return config


class UnifiedModelLoader:
    """Universal model loader for all RingRift neural network architectures.

    Provides a single entry point for loading ANY model checkpoint,
    automatically detecting architecture, inferring configuration, and
    instantiating the correct model class.

    Features:
    - Automatic architecture detection from checkpoint keys
    - Configuration inference from tensor shapes
    - Graceful fallback chain: specific model -> fresh weights -> heuristic
    - Model caching to avoid redundant loads
    - Device management (CPU/CUDA/MPS)

    Usage:
        loader = UnifiedModelLoader()
        loaded = loader.load("models/my_model.pth", BoardType.SQUARE8)
        ai = UniversalAI.from_loaded_model(loaded, player_number=1)
    """

    # Class-level cache
    _cache: dict[str, LoadedModel] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        device: str | torch.device | None = None,
        cache_enabled: bool = True,
        max_cache_size: int = 10,
    ):
        """Initialize the loader.

        Args:
            device: Target device for models. None = auto-detect.
            cache_enabled: Whether to cache loaded models.
            max_cache_size: Maximum models to keep in cache.
        """
        self.device = self._resolve_device(device)
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        """Resolve device specification to torch.device."""
        if device is not None:
            return torch.device(device) if isinstance(device, str) else device

        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load(
        self,
        checkpoint_path: str | Path,
        board_type: BoardType | None = None,
        num_players: int | None = None,
        strict: bool = False,
        allow_fresh: bool = True,
    ) -> LoadedModel:
        """Load a model from checkpoint with automatic architecture detection.

        Args:
            checkpoint_path: Path to the .pth/.pt checkpoint file.
            board_type: Expected board type (for validation/fallback).
            num_players: Expected number of players.
            strict: If True, fail on any mismatch. If False, use best effort.
            allow_fresh: If True, return fresh weights if load fails.

        Returns:
            LoadedModel containing the model and metadata.

        Raises:
            FileNotFoundError: If checkpoint doesn't exist and allow_fresh=False.
        """
        path = Path(checkpoint_path)
        cache_key = f"{path.resolve()}:{self.device}"

        # Check cache first
        if self.cache_enabled:
            with self._cache_lock:
                if cache_key in self._cache:
                    logger.debug(f"Cache hit for {path}")
                    return self._cache[cache_key]

        # Load checkpoint
        if not path.exists():
            if not allow_fresh:
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            logger.warning(f"Checkpoint not found: {path}, creating fresh model")
            return self._create_fresh_model(board_type, num_players)

        try:
            from app.utils.torch_utils import safe_load_checkpoint

            checkpoint = safe_load_checkpoint(
                str(path),
                map_location=self.device,
                warn_on_unsafe=False,
            )
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {path}: {e}")
            if not allow_fresh:
                raise
            return self._create_fresh_model(board_type, num_players)

        # Extract state dict and metadata
        metadata = {}
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            metadata = checkpoint.get("_versioning_metadata", {})
            if not metadata:
                metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
        else:
            state_dict = checkpoint

        # Detect architecture
        architecture = detect_architecture(state_dict)
        if architecture == ModelArchitecture.UNKNOWN:
            logger.warning(f"Unknown architecture in {path}, attempting best-effort load")

        # Infer configuration
        config = infer_config_from_checkpoint(state_dict, architecture, metadata)

        # Override with provided values if available
        if board_type is not None:
            config.board_type = board_type
        if num_players is not None:
            config.num_players = num_players

        # Ensure board_type has a default
        if config.board_type is None:
            config.board_type = BoardType.SQUARE8

        # VALIDATION: Check if policy_size matches expected board-aware encoding
        # This catches models trained with legacy_max_n encoding (~59K for square8)
        expected_policy_size = POLICY_SIZE_MAP.get(config.board_type, 7000)
        if config.policy_size > expected_policy_size * 2:
            # Policy size is way too large - likely legacy_max_n encoding
            logger.warning(
                f"CHECKPOINT POLICY SIZE MISMATCH!\n"
                f"  Checkpoint: {path}\n"
                f"  Inferred policy_size: {config.policy_size}\n"
                f"  Expected for {config.board_type.value}: {expected_policy_size}\n\n"
                f"This model was likely trained with legacy_max_n encoding.\n"
                f"It will produce incorrect predictions. Retrain with board-aware data."
            )
            # For strict mode, fail immediately
            if strict:
                raise ValueError(
                    f"Policy size mismatch: {config.policy_size} > {expected_policy_size} "
                    f"for {config.board_type.value}. Model was trained with deprecated "
                    f"legacy_max_n encoding and will not work correctly."
                )

        # Instantiate model
        try:
            model = self._instantiate_model(architecture, config)
        except Exception as e:
            logger.warning(f"Failed to instantiate model for {architecture}: {e}")
            if not allow_fresh:
                raise
            return self._create_fresh_model(board_type, num_players)

        # Apply NNUE-specific migration if needed (V1 -> V2 feature padding)
        if architecture in {ModelArchitecture.NNUE_VALUE_ONLY, ModelArchitecture.NNUE_WITH_POLICY}:
            from app.ai.nnue import (
                _migrate_legacy_state_dict,
                FEATURE_DIMS,
            )
            arch_version = metadata.get("architecture_version", "v1.0.0")
            target_input_size = FEATURE_DIMS.get(config.board_type, FEATURE_DIMS[BoardType.SQUARE8])
            state_dict, detected_version = _migrate_legacy_state_dict(
                state_dict, arch_version, target_input_size, config.board_type
            )

        # Load weights
        try:
            model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            if strict:
                raise
            logger.warning(f"Partial weight load for {path}: {e}")
            try:
                model.load_state_dict(state_dict, strict=False)
            except Exception as e2:
                logger.warning(f"Even non-strict load failed: {e2}")
                if not allow_fresh:
                    raise
                return self._create_fresh_model(board_type, num_players)

        model = model.to(self.device)
        model.eval()

        # Determine capabilities
        has_policy = architecture not in {ModelArchitecture.NNUE_VALUE_ONLY}
        is_multiplayer = architecture in {
            ModelArchitecture.CNN_V2,
            ModelArchitecture.CNN_V2_LITE,
            ModelArchitecture.CNN_V3,
            ModelArchitecture.CNN_V3_LITE,
            ModelArchitecture.CNN_V4,
            ModelArchitecture.HEX_V2,
            ModelArchitecture.HEX_V2_LITE,
            ModelArchitecture.HEX_V3,
            ModelArchitecture.HEX_V3_LITE,
        }

        # Create result
        loaded = LoadedModel(
            model=model,
            architecture=architecture,
            config=config,
            checkpoint_path=path,
            device=self.device,
            has_policy=has_policy,
            has_value=True,
            is_multiplayer_value=is_multiplayer,
        )

        # Cache if enabled
        if self.cache_enabled:
            with self._cache_lock:
                if len(self._cache) >= self.max_cache_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                self._cache[cache_key] = loaded

        logger.info(
            f"Loaded {architecture.name} model from {path} "
            f"(board={config.board_type}, players={config.num_players})"
        )

        return loaded

    def _instantiate_model(
        self,
        architecture: ModelArchitecture,
        config: InferredModelConfig,
    ) -> nn.Module:
        """Instantiate the correct model class for the architecture."""
        board_size = self._board_size(config.board_type)

        if architecture == ModelArchitecture.NNUE_VALUE_ONLY:
            from app.ai.nnue import RingRiftNNUE

            return RingRiftNNUE(
                board_type=config.board_type or BoardType.SQUARE8,
                hidden_dim=config.hidden_dim,
            )

        elif architecture == ModelArchitecture.NNUE_WITH_POLICY:
            from app.ai.nnue_policy import RingRiftNNUEWithPolicy

            return RingRiftNNUEWithPolicy(
                board_type=config.board_type or BoardType.SQUARE8,
                hidden_dim=config.hidden_dim,
            )

        elif architecture in {ModelArchitecture.CNN_V2, ModelArchitecture.CNN_V2_LITE}:
            from app.ai.neural_net.square_architectures import (
                RingRiftCNN_v2,
                RingRiftCNN_v2_Lite,
            )

            cls = RingRiftCNN_v2_Lite if config.is_lite_variant else RingRiftCNN_v2
            return cls(
                board_size=board_size,
                in_channels=config.input_channels,
                global_features=config.global_features,
                num_res_blocks=config.num_res_blocks,
                num_filters=config.num_filters,
                history_length=config.history_length,
                num_players=config.num_players,
                policy_size=config.policy_size,
                policy_intermediate=config.policy_intermediate,
                value_intermediate=config.value_intermediate,
            )

        elif architecture in {ModelArchitecture.CNN_V3, ModelArchitecture.CNN_V3_LITE}:
            from app.ai.neural_net.square_architectures import (
                RingRiftCNN_v3,
                RingRiftCNN_v3_Lite,
            )

            # V3 uses spatial policy heads (like V4), not FC policy heads
            # so it doesn't use policy_intermediate
            cls = RingRiftCNN_v3_Lite if config.is_lite_variant else RingRiftCNN_v3
            return cls(
                board_size=board_size,
                in_channels=config.input_channels,
                global_features=config.global_features,
                num_res_blocks=config.num_res_blocks,
                num_filters=config.num_filters,
                history_length=config.history_length,
                num_players=config.num_players,
                policy_size=config.policy_size,
                value_intermediate=config.value_intermediate,
            )

        elif architecture == ModelArchitecture.CNN_V4:
            from app.ai.neural_net.square_architectures import RingRiftCNN_v4

            # V4 has different constructor signature than V2/V3:
            # - No policy_intermediate (uses fixed spatial policy)
            # - Has initial_kernel_size, num_attention_heads, dropout
            return RingRiftCNN_v4(
                board_size=board_size,
                in_channels=config.input_channels,
                global_features=config.global_features,
                num_res_blocks=config.num_res_blocks,
                num_filters=config.num_filters,
                history_length=config.history_length,
                num_players=config.num_players,
                policy_size=config.policy_size,
                value_intermediate=config.value_intermediate,
            )

        elif architecture in {ModelArchitecture.HEX_V2, ModelArchitecture.HEX_V2_LITE}:
            from app.ai.neural_net.hex_architectures import (
                HexNeuralNet_v2,
                HexNeuralNet_v2_Lite,
            )

            cls = HexNeuralNet_v2_Lite if config.is_lite_variant else HexNeuralNet_v2
            # V2 uses board_size (2*hex_radius+1) as the primary spatial dimension
            hex_board_size = 2 * config.hex_radius + 1
            # Hex models use total input channels (base * (history+1))
            total_in_channels = config.input_channels
            if config.history_length > 0:
                total_in_channels = config.input_channels * (config.history_length + 1)
            return cls(
                board_size=hex_board_size,
                hex_radius=config.hex_radius,
                in_channels=total_in_channels,
                global_features=config.global_features,
                num_res_blocks=config.num_res_blocks,
                num_filters=config.num_filters,
                num_players=config.num_players,
                policy_size=config.policy_size,
            )

        elif architecture in {ModelArchitecture.HEX_V3, ModelArchitecture.HEX_V3_LITE}:
            from app.ai.neural_net.hex_architectures import (
                HexNeuralNet_v3,
                HexNeuralNet_v3_Lite,
            )

            cls = HexNeuralNet_v3_Lite if config.is_lite_variant else HexNeuralNet_v3
            # V3 uses board_size (2*hex_radius+1) as the primary spatial dimension
            hex_board_size = 2 * config.hex_radius + 1
            # Hex models use total input channels (base * (history+1))
            # If history_length is inferred, compute total; otherwise use as-is
            total_in_channels = config.input_channels
            if config.history_length > 0:
                total_in_channels = config.input_channels * (config.history_length + 1)
            return cls(
                board_size=hex_board_size,
                hex_radius=config.hex_radius,
                in_channels=total_in_channels,
                global_features=config.global_features,
                num_res_blocks=config.num_res_blocks,
                num_filters=config.num_filters,
                num_players=config.num_players,
                policy_size=config.policy_size,
            )

        elif architecture == ModelArchitecture.UNKNOWN:
            # Try CNN v2 as fallback
            from app.ai.neural_net.square_architectures import RingRiftCNN_v2

            return RingRiftCNN_v2(
                board_size=board_size,
                in_channels=config.input_channels,
                global_features=config.global_features,
                num_res_blocks=config.num_res_blocks,
                num_filters=config.num_filters,
                history_length=config.history_length,
                num_players=config.num_players,
                policy_size=config.policy_size,
                policy_intermediate=config.policy_intermediate,
                value_intermediate=config.value_intermediate,
            )

        raise ValueError(f"Unsupported architecture: {architecture}")

    def _create_fresh_model(
        self,
        board_type: BoardType | None,
        num_players: int | None,
    ) -> LoadedModel:
        """Create a model with fresh (random) weights."""
        bt = board_type or BoardType.SQUARE8
        np_ = num_players or 2

        from app.ai.neural_net.model_factory import create_model

        model = create_model(board_type=bt, num_players=np_)
        model = model.to(self.device)
        model.eval()

        config = InferredModelConfig(
            architecture=ModelArchitecture.CNN_V2,
            board_type=bt,
            num_players=np_,
            num_res_blocks=12,
            num_filters=192,
            policy_size=POLICY_SIZE_MAP.get(bt, 4672),
            is_lite_variant=False,
            hidden_dim=256,
            input_channels=56,
            global_features=20,
        )

        return LoadedModel(
            model=model,
            architecture=ModelArchitecture.CNN_V2,
            config=config,
            checkpoint_path=Path("/dev/null"),
            device=self.device,
            has_policy=True,
            has_value=True,
            is_multiplayer_value=True,
        )

    @staticmethod
    def _board_size(board_type: BoardType | None) -> int:
        """Get spatial size for board type."""
        sizes = {
            BoardType.SQUARE8: 8,
            BoardType.SQUARE19: 19,
            BoardType.HEX8: 9,
            BoardType.HEXAGONAL: 25,
        }
        return sizes.get(board_type or BoardType.SQUARE8, 8)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the model cache and free memory."""
        with cls._cache_lock:
            for loaded in cls._cache.values():
                try:
                    loaded.model.cpu()
                except Exception:
                    pass
            cls._cache.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
