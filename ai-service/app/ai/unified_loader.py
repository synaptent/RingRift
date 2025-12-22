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


# Board type to policy size mapping
POLICY_SIZE_MAP = {
    BoardType.SQUARE8: 4672,
    BoardType.SQUARE19: 43681,
    BoardType.HEX8: 13953,
    BoardType.HEXAGONAL: 55809,
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
    input_channels: int = 56
    global_features: int = 20
    hex_radius: int = 4


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
                    arch = ModelArchitecture.HEX_V2
                else:
                    arch = ModelArchitecture.CNN_V2

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

    # Infer from conv1 shape: [out_channels, in_channels, H, W]
    if "conv1.weight" in state_dict:
        conv1_shape = state_dict["conv1.weight"].shape
        config.num_filters = conv1_shape[0]
        config.input_channels = conv1_shape[1]

        # Lite variant detection: fewer filters
        if config.num_filters <= 96:
            config.is_lite_variant = True

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
    if "value_fc2.weight" in state_dict:
        config.num_players = state_dict["value_fc2.weight"].shape[0]

    # NNUE-specific: hidden_dim from accumulator
    if "accumulator.weight" in state_dict:
        config.hidden_dim = state_dict["accumulator.weight"].shape[0]

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

        # Instantiate model
        try:
            model = self._instantiate_model(architecture, config)
        except Exception as e:
            logger.warning(f"Failed to instantiate model for {architecture}: {e}")
            if not allow_fresh:
                raise
            return self._create_fresh_model(board_type, num_players)

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
                total_in_channels=config.input_channels,
                global_features=config.global_features,
                num_res_blocks=config.num_res_blocks,
                num_filters=config.num_filters,
                num_players=config.num_players,
                policy_size=config.policy_size,
            )

        elif architecture in {ModelArchitecture.CNN_V3, ModelArchitecture.CNN_V3_LITE}:
            from app.ai.neural_net.square_architectures import (
                RingRiftCNN_v3,
                RingRiftCNN_v3_Lite,
            )

            cls = RingRiftCNN_v3_Lite if config.is_lite_variant else RingRiftCNN_v3
            return cls(
                board_size=board_size,
                total_in_channels=config.input_channels,
                global_features=config.global_features,
                num_res_blocks=config.num_res_blocks,
                num_filters=config.num_filters,
                num_players=config.num_players,
                policy_size=config.policy_size,
            )

        elif architecture == ModelArchitecture.CNN_V4:
            from app.ai.neural_net.square_architectures import RingRiftCNN_v4

            return RingRiftCNN_v4(
                board_size=board_size,
                total_in_channels=config.input_channels,
                global_features=config.global_features,
                num_res_blocks=config.num_res_blocks,
                num_filters=config.num_filters,
                num_players=config.num_players,
                policy_size=config.policy_size,
            )

        elif architecture in {ModelArchitecture.HEX_V2, ModelArchitecture.HEX_V2_LITE}:
            from app.ai.neural_net.hex_architectures import (
                HexNeuralNet_v2,
                HexNeuralNet_v2_Lite,
            )

            cls = HexNeuralNet_v2_Lite if config.is_lite_variant else HexNeuralNet_v2
            return cls(
                hex_radius=config.hex_radius,
                total_in_channels=config.input_channels,
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
            return cls(
                hex_radius=config.hex_radius,
                total_in_channels=config.input_channels,
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
                total_in_channels=config.input_channels,
                global_features=config.global_features,
                num_res_blocks=config.num_res_blocks,
                num_filters=config.num_filters,
                num_players=config.num_players,
                policy_size=config.policy_size,
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
