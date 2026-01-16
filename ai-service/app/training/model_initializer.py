"""Model initialization for RingRift training.

December 2025: Extracted from train.py to improve modularity.

This module provides the ModelInitializer class which handles all aspects
of model creation, weight loading, and DDP wrapping for training.

Usage:
    from app.training.model_initializer import ModelInitializer, ModelConfig

    config = ModelConfig(
        board_type=BoardType.HEX8,
        num_players=2,
        model_version="v3",
    )
    initializer = ModelInitializer(config, device)
    result = initializer.create_model()
    model = result.model
"""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from app.utils.numpy_utils import safe_load_npz
from app.utils.torch_utils import safe_load_checkpoint

if TYPE_CHECKING:
    from app.coordination.types import BoardType

logger = logging.getLogger(__name__)

# Board size constants
HEX_BOARD_SIZE = 25  # 2 * 12 + 1 for radius 12
HEX8_BOARD_SIZE = 9  # 2 * 4 + 1 for radius 4
MAX_PLAYERS = 4


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    board_type: "BoardType"
    num_players: int = 2
    multi_player: bool = False
    model_version: str = "v2"
    model_type: str = "cnn"  # "cnn", "gnn", or "hybrid"
    num_res_blocks: int | None = None
    num_filters: int | None = None
    history_length: int = 3
    dropout: float = 0.08
    freeze_policy: bool = False
    feature_version: int = 1

    # Inferred from data
    policy_size: int = 0
    detected_num_heuristics: int | None = None

    # Model ID for naming
    model_id: str = ""


@dataclass
class DatasetMetadata:
    """Metadata extracted from NPZ training data."""

    in_channels: int | None = None
    globals_dim: int | None = None
    policy_encoding: str | None = None
    history_length: int | None = None
    feature_version: int | None = None
    inferred_policy_size: int | None = None
    num_heuristics: int | None = None


@dataclass
class ModelInitResult:
    """Result of model initialization."""

    model: nn.Module
    model_version: str
    policy_size: int
    board_size: int
    effective_blocks: int
    effective_filters: int
    feature_version: int
    is_hex_model: bool = False
    hex_radius: int | None = None
    errors: list[str] = field(default_factory=list)


class ModelInitializer:
    """Creates and initializes models for training.

    Handles:
    - Board size determination
    - Model version selection
    - Dataset metadata extraction
    - Model creation
    - Weight loading (transfer learning and resume)
    - DDP wrapping

    Example:
        initializer = ModelInitializer(config, device)
        result = initializer.create_model()
        model = result.model
    """

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device,
        distributed: bool = False,
        is_main_process: bool = True,
    ):
        """Initialize the model initializer.

        Args:
            config: Model configuration
            device: Target device for model
            distributed: Whether distributed training is enabled
            is_main_process: Whether this is the main process
        """
        self.config = config
        self.device = device
        self.distributed = distributed
        self.is_main_process = is_main_process

    def create_model(
        self,
        data_path: str | list[str] | None = None,
    ) -> ModelInitResult:
        """Create model based on configuration.

        Args:
            data_path: Optional path(s) to training data for metadata extraction

        Returns:
            ModelInitResult with initialized model
        """
        # Determine board size
        board_size = self._get_board_size()
        is_hex_model = self._is_hex_model()
        hex_radius = self._get_hex_radius() if is_hex_model else None

        # Extract dataset metadata if data path provided
        metadata = None
        if data_path:
            data_path_str = data_path[0] if isinstance(data_path, list) else data_path
            if data_path_str and os.path.exists(data_path_str):
                metadata = self._extract_dataset_metadata(data_path_str)
                self._validate_metadata(metadata, data_path_str)

        # Determine policy size
        policy_size = self._determine_policy_size(metadata, board_size, is_hex_model)

        # Determine effective blocks and filters
        effective_blocks, effective_filters = self._get_effective_architecture()

        # Detect heuristics count from metadata
        detected_num_heuristics = None
        if metadata and metadata.num_heuristics:
            detected_num_heuristics = metadata.num_heuristics

        # Create model
        model = self._create_model_instance(
            board_size=board_size,
            policy_size=policy_size,
            effective_blocks=effective_blocks,
            effective_filters=effective_filters,
            is_hex_model=is_hex_model,
            hex_radius=hex_radius,
            detected_num_heuristics=detected_num_heuristics,
        )

        # Set feature version
        feature_version = self.config.feature_version
        if metadata and metadata.feature_version:
            feature_version = metadata.feature_version
        with contextlib.suppress(Exception):
            model.feature_version = feature_version

        # Move to device
        model.to(self.device)

        # Validate value head
        self._validate_value_head(model, "after model creation")

        # Apply freeze_policy if requested
        if self.config.freeze_policy:
            self._apply_freeze_policy(model)

        return ModelInitResult(
            model=model,
            model_version=self.config.model_version,
            policy_size=policy_size,
            board_size=board_size,
            effective_blocks=effective_blocks,
            effective_filters=effective_filters,
            feature_version=feature_version,
            is_hex_model=is_hex_model,
            hex_radius=hex_radius,
        )

    def load_init_weights(
        self,
        model: nn.Module,
        init_weights_path: str,
        strict: bool = False,
    ) -> dict[str, list[str]]:
        """Load initial weights for transfer learning.

        Args:
            model: Model to load weights into
            init_weights_path: Path to weights file
            strict: Whether to require exact match

        Returns:
            Dict with 'missing_keys' and 'unexpected_keys' lists
        """
        if not os.path.exists(init_weights_path):
            raise FileNotFoundError(f"Init weights not found: {init_weights_path}")

        try:
            from app.training.checkpointing import load_weights_only

            load_result = load_weights_only(
                init_weights_path,
                model,
                device=self.device,
                strict=strict,
            )

            if self.is_main_process:
                logger.info(f"Loaded initial weights from {init_weights_path}")
                if load_result.get("missing_keys"):
                    logger.info(
                        f"  Missing keys (randomly initialized): "
                        f"{len(load_result['missing_keys'])}"
                    )
                if load_result.get("unexpected_keys"):
                    logger.info(
                        f"  Unexpected keys (ignored): "
                        f"{len(load_result['unexpected_keys'])}"
                    )

            # Validate value head after loading
            self._validate_value_head(model, "after loading init_weights")

            return load_result

        except (OSError, RuntimeError, ValueError, KeyError) as e:
            if self.is_main_process:
                logger.warning(
                    f"Could not load init weights from {init_weights_path}: {e}"
                )
            raise

    def load_checkpoint_weights(
        self,
        model: nn.Module,
        checkpoint_path: str,
    ) -> None:
        """Load existing checkpoint weights.

        Args:
            model: Model to load weights into
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            if self.is_main_process:
                logger.debug(f"No existing checkpoint at {checkpoint_path}")
            return

        try:
            checkpoint = safe_load_checkpoint(
                checkpoint_path,
                map_location=self.device,
                warn_on_unsafe=False,
            )

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            if self.is_main_process:
                logger.info(f"Loaded existing model weights from {checkpoint_path}")

            # Validate value head after loading
            self._validate_value_head(model, "after loading checkpoint")

        except (OSError, RuntimeError, ValueError, KeyError) as e:
            if self.is_main_process:
                logger.warning(f"Could not load existing weights: {e}. Starting fresh.")

    def wrap_ddp(
        self,
        model: nn.Module,
        find_unused_parameters: bool = False,
    ) -> nn.Module:
        """Wrap model with DistributedDataParallel.

        Args:
            model: Model to wrap
            find_unused_parameters: Enable unused parameter detection

        Returns:
            DDP-wrapped model
        """
        if not self.distributed:
            return model

        try:
            from app.training.distributed import wrap_model_ddp

            model = wrap_model_ddp(
                model,
                self.device,
                find_unused_parameters=find_unused_parameters,
            )

            if self.is_main_process:
                logger.info("Model wrapped with DistributedDataParallel")

            return model

        except ImportError:
            logger.warning("Distributed module not available, skipping DDP wrap")
            return model

    def _get_board_size(self) -> int:
        """Get the spatial board size based on board type."""
        from app.coordination.types import BoardType

        if self.config.board_type == BoardType.SQUARE19:
            return 19
        elif self.config.board_type == BoardType.HEXAGONAL:
            return HEX_BOARD_SIZE
        elif self.config.board_type == BoardType.HEX8:
            return HEX8_BOARD_SIZE
        else:
            return 8  # Default square8

    def _is_hex_model(self) -> bool:
        """Check if this is a hexagonal board model."""
        from app.coordination.types import BoardType

        return self.config.board_type in (BoardType.HEXAGONAL, BoardType.HEX8)

    def _get_hex_radius(self) -> int:
        """Get the hex board radius."""
        from app.coordination.types import BoardType

        if self.config.board_type == BoardType.HEX8:
            return 4
        elif self.config.board_type == BoardType.HEXAGONAL:
            return 12
        return 4  # Default

    def _get_effective_architecture(self) -> tuple[int, int]:
        """Get effective blocks and filters from config or defaults."""
        blocks = self.config.num_res_blocks
        filters = self.config.num_filters

        if blocks is None:
            blocks = 6  # Default
        if filters is None:
            filters = 96  # Default

        return blocks, filters

    def _extract_dataset_metadata(self, data_path: str) -> DatasetMetadata:
        """Extract metadata from NPZ training data.

        Args:
            data_path: Path to NPZ file

        Returns:
            DatasetMetadata with extracted values
        """
        metadata = DatasetMetadata()

        try:
            with safe_load_npz(data_path, mmap_mode="r") as d:
                # Feature channels
                if "features" in d:
                    feat_shape = d["features"].shape
                    if len(feat_shape) >= 2:
                        metadata.in_channels = int(feat_shape[1])

                # Globals dimension
                if "globals" in d:
                    glob_shape = d["globals"].shape
                    if len(glob_shape) >= 2:
                        metadata.globals_dim = int(glob_shape[1])

                # Policy encoding
                if "policy_encoding" in d:
                    try:
                        metadata.policy_encoding = str(
                            np.asarray(d["policy_encoding"]).item()
                        )
                    except (ValueError, TypeError, AttributeError):
                        pass

                # History length
                if "history_length" in d:
                    try:
                        metadata.history_length = int(
                            np.asarray(d["history_length"]).item()
                        )
                    except (ValueError, TypeError, AttributeError):
                        pass

                # Feature version
                if "feature_version" in d:
                    try:
                        metadata.feature_version = int(
                            np.asarray(d["feature_version"]).item()
                        )
                    except (ValueError, TypeError, AttributeError):
                        pass

                # Infer policy size from policy_indices
                if "policy_indices" in d:
                    pi = d["policy_indices"]
                    max_idx = -1
                    for i in range(min(len(pi), 10000)):  # Sample first 10k
                        arr = np.asarray(pi[i])
                        if arr.size == 0:
                            continue
                        local_max = int(arr.max())
                        if local_max > max_idx:
                            max_idx = local_max
                    if max_idx >= 0:
                        metadata.inferred_policy_size = max_idx + 1

                # Heuristics count (for v5-heavy and v5-heavy-large/xl models)
                if "heuristics" in d:
                    heur_shape = d["heuristics"].shape
                    if len(heur_shape) >= 2:
                        metadata.num_heuristics = int(heur_shape[1])

        except (OSError, KeyError, ValueError, AttributeError) as e:
            if self.is_main_process:
                logger.warning(f"Failed to extract metadata from {data_path}: {e}")

        return metadata

    def _validate_metadata(
        self,
        metadata: DatasetMetadata,
        data_path: str,
    ) -> None:
        """Validate dataset metadata against config.

        Args:
            metadata: Extracted metadata
            data_path: Path to data file (for error messages)

        Raises:
            ValueError: If metadata doesn't match config
        """
        # Validate history length
        if (
            metadata.history_length is not None
            and metadata.history_length != self.config.history_length
        ):
            raise ValueError(
                f"Training history_length does not match dataset metadata.\n"
                f"  dataset={data_path}\n"
                f"  dataset_history_length={metadata.history_length}\n"
                f"  config.history_length={self.config.history_length}\n"
                "Regenerate the dataset with matching --history-length."
            )

        # Validate feature version
        if (
            metadata.feature_version is not None
            and metadata.feature_version != self.config.feature_version
        ):
            raise ValueError(
                f"Training feature_version does not match dataset metadata.\n"
                f"  dataset={data_path}\n"
                f"  dataset_feature_version={metadata.feature_version}\n"
                f"  config_feature_version={self.config.feature_version}\n"
                "Regenerate the dataset with matching --feature-version."
            )

        # Validate globals dimension
        if metadata.globals_dim is None:
            raise ValueError(
                f"Dataset is missing globals features required for training.\n"
                f"  dataset={data_path}\n"
                "Regenerate the dataset with scripts/export_replay_dataset.py."
            )

    def _determine_policy_size(
        self,
        metadata: DatasetMetadata | None,
        board_size: int,
        is_hex_model: bool,
    ) -> int:
        """Determine the policy head size.

        Args:
            metadata: Dataset metadata (may have inferred size)
            board_size: Board spatial size
            is_hex_model: Whether this is a hex model

        Returns:
            Policy size for the model
        """
        if self.config.policy_size > 0:
            return self.config.policy_size

        if metadata and metadata.inferred_policy_size:
            return metadata.inferred_policy_size

        # Default fallback based on board type
        # Use canonical constants from neural_net module
        from app.ai.neural_net.constants import POLICY_SIZE_HEX8, P_HEX

        if is_hex_model:
            if board_size == HEX8_BOARD_SIZE:
                return POLICY_SIZE_HEX8  # 4500
            return P_HEX  # 91876
        else:
            return board_size * board_size * 2  # square board default

    def _create_model_instance(
        self,
        board_size: int,
        policy_size: int,
        effective_blocks: int,
        effective_filters: int,
        is_hex_model: bool,
        hex_radius: int | None,
        detected_num_heuristics: int | None,
    ) -> nn.Module:
        """Create the actual model instance.

        Args:
            board_size: Board spatial size
            policy_size: Policy head size
            effective_blocks: Number of residual blocks
            effective_filters: Number of filters
            is_hex_model: Whether this is a hex model
            hex_radius: Hex board radius (if hex)
            detected_num_heuristics: Detected heuristics count from NPZ

        Returns:
            Initialized model
        """
        version = self.config.model_version
        num_players = self.config.num_players
        dropout = self.config.dropout
        history_length = self.config.history_length

        # Calculate input channels
        if is_hex_model:
            # Hex models use 16 channels per frame
            hex_in_channels = 16 * (history_length + 1)
            hex_num_players = MAX_PLAYERS if self.config.multi_player else num_players
        else:
            # Square models use 14 channels per frame
            pass

        # Model type selection
        if self.config.model_type in ("gnn", "hybrid"):
            return self._create_gnn_model(num_players, effective_filters)

        # Version-specific model creation
        if is_hex_model:
            return self._create_hex_model(
                version=version,
                board_size=board_size,
                hex_radius=hex_radius,
                hex_in_channels=hex_in_channels,
                hex_num_players=hex_num_players,
                policy_size=policy_size,
                effective_blocks=effective_blocks,
                effective_filters=effective_filters,
                dropout=dropout,
                detected_num_heuristics=detected_num_heuristics,
            )
        else:
            return self._create_square_model(
                version=version,
                board_size=board_size,
                policy_size=policy_size,
                num_players=num_players,
                effective_blocks=effective_blocks,
                effective_filters=effective_filters,
                history_length=history_length,
                dropout=dropout,
                detected_num_heuristics=detected_num_heuristics,
            )

    def _create_gnn_model(
        self,
        num_players: int,
        effective_filters: int,
    ) -> nn.Module:
        """Create GNN/Hybrid model."""
        from app.ai.neural_net.model_factory import create_model_for_board, HAS_GNN

        if not HAS_GNN:
            raise ImportError(
                f"Model type '{self.config.model_type}' requires PyTorch Geometric. "
                "Install with: pip install torch-geometric torch-scatter torch-sparse"
            )

        gnn_num_players = MAX_PLAYERS if self.config.multi_player else num_players
        model = create_model_for_board(
            board_type=self.config.board_type,
            memory_tier=self.config.model_type,
            num_players=gnn_num_players,
        )

        if self.is_main_process:
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(
                f"Initialized {self.config.model_type.upper()} model for "
                f"{self.config.board_type.name}: {param_count:,} parameters"
            )

        return model

    def _create_hex_model(
        self,
        version: str,
        board_size: int,
        hex_radius: int,
        hex_in_channels: int,
        hex_num_players: int,
        policy_size: int,
        effective_blocks: int,
        effective_filters: int,
        dropout: float,
        detected_num_heuristics: int | None,
    ) -> nn.Module:
        """Create hexagonal board model."""
        if version in ("v5", "v5-heavy", "v5-gnn"):
            from app.ai.neural_net.v5_heavy import (
                HexNeuralNet_v5_Heavy,
                NUM_HEURISTIC_FEATURES_FAST,
            )

            v5_filters = self.config.num_filters or 160
            use_gnn = version == "v5-gnn"
            num_heuristics = detected_num_heuristics or NUM_HEURISTIC_FEATURES_FAST

            model = HexNeuralNet_v5_Heavy(
                board_size=board_size,
                hex_radius=hex_radius,
                in_channels=hex_in_channels,
                global_features=20,
                num_filters=v5_filters,
                policy_size=policy_size,
                num_players=hex_num_players,
                num_heuristics=num_heuristics,
                use_gnn=use_gnn,
                dropout=dropout,
            )

            if self.is_main_process:
                heuristic_mode = "full (49)" if num_heuristics >= 49 else "fast (21)"
                logger.info(
                    f"Initializing HexNeuralNet_v5_Heavy: board_size={board_size}, "
                    f"policy_size={policy_size}, heuristics={heuristic_mode}"
                )

            return model

        elif version == "v4":
            from app.ai.neural_net.hex_architectures import HexNeuralNet_v4

            # V4 computes policy_size dynamically from board_size when None
            return HexNeuralNet_v4(
                in_channels=hex_in_channels,
                global_features=20,
                num_res_blocks=effective_blocks,
                num_filters=effective_filters,
                board_size=board_size,
                hex_radius=hex_radius,
                policy_size=None,  # Computed dynamically from board_size
                num_players=hex_num_players,
            )

        elif version in ("v3", "v3-flat"):
            if version == "v3-flat":
                from app.ai.neural_net.hex_v3 import HexNeuralNet_v3_Flat

                return HexNeuralNet_v3_Flat(
                    in_channels=hex_in_channels,
                    global_features=20,
                    num_res_blocks=effective_blocks,
                    num_filters=effective_filters,
                    board_size=board_size,
                    hex_radius=hex_radius,
                    policy_size=policy_size,
                    num_players=hex_num_players,
                )
            else:
                from app.ai.neural_net.hex_v3 import HexNeuralNet_v3

                return HexNeuralNet_v3(
                    in_channels=hex_in_channels,
                    global_features=20,
                    num_res_blocks=effective_blocks,
                    num_filters=effective_filters,
                    board_size=board_size,
                    hex_radius=hex_radius,
                    policy_size=policy_size,
                    num_players=hex_num_players,
                )

        else:  # v2 default
            from app.ai.neural_net.hex_v2 import HexNeuralNet_v2

            return HexNeuralNet_v2(
                in_channels=hex_in_channels,
                global_features=20,
                num_res_blocks=effective_blocks,
                num_filters=effective_filters,
                board_size=board_size,
                hex_radius=hex_radius,
                policy_size=policy_size,
                num_players=hex_num_players,
            )

    def _create_square_model(
        self,
        version: str,
        board_size: int,
        policy_size: int,
        num_players: int,
        effective_blocks: int,
        effective_filters: int,
        history_length: int,
        dropout: float,
        detected_num_heuristics: int | None,
    ) -> nn.Module:
        """Create square board model."""
        model_num_players = MAX_PLAYERS if self.config.multi_player else num_players

        if version in ("v5-heavy-large", "v5-heavy-xl", "v6", "v6-xl"):
            # V5 Heavy Large: scaled-up v5-heavy for 2000+ Elo
            # Note: "v6" and "v6-xl" are deprecated aliases for backward compatibility
            from app.ai.neural_net.v5_heavy_large import create_v5_heavy_large

            # Map version to variant
            variant_map = {"v5-heavy-xl": "xl", "v6-xl": "xl"}
            large_variant = variant_map.get(version, "large")
            num_heuristics = detected_num_heuristics or 49

            model = create_v5_heavy_large(
                board_type=self.config.board_type.name.lower(),
                num_players=model_num_players,
                variant=large_variant,
                num_heuristics=num_heuristics,
                dropout=dropout,
            )

            if self.is_main_process:
                param_count = sum(p.numel() for p in model.parameters())
                display_name = f"V5-Heavy-{large_variant.upper()}"
                logger.info(
                    f"Initializing {display_name} model: {param_count:,} parameters"
                )

            return model

        elif version in ("v5", "v5-heavy"):
            from app.ai.neural_net.v5_heavy import (
                RingRiftCNN_v5_Heavy,
                NUM_HEURISTIC_FEATURES_FAST,
            )

            v5_filters = self.config.num_filters or 160
            use_gnn = version == "v5-gnn"
            num_heuristics = detected_num_heuristics or NUM_HEURISTIC_FEATURES_FAST

            model = RingRiftCNN_v5_Heavy(
                board_size=board_size,
                in_channels=14,
                global_features=20,
                history_length=history_length,
                policy_size=policy_size,
                num_players=model_num_players,
                num_filters=v5_filters,
                num_heuristics=num_heuristics,
                use_gnn=use_gnn,
                dropout=dropout,
            )

            if self.is_main_process:
                heuristic_mode = "full (49)" if num_heuristics >= 49 else "fast (21)"
                logger.info(
                    f"Initializing RingRiftCNN_v5_Heavy: board_size={board_size}, "
                    f"policy_size={policy_size}, heuristics={heuristic_mode}"
                )

            return model

        elif version in ("v3", "v3-flat"):
            if version == "v3-flat":
                from app.ai.neural_net.cnn_v3 import RingRiftCNN_v3_Flat

                return RingRiftCNN_v3_Flat(
                    board_size=board_size,
                    in_channels=14,
                    global_features=20,
                    history_length=history_length,
                    policy_size=policy_size,
                    num_players=model_num_players,
                    num_res_blocks=effective_blocks,
                    num_filters=effective_filters,
                )
            else:
                from app.ai.neural_net.cnn_v3 import RingRiftCNN_v3

                model = RingRiftCNN_v3(
                    board_size=board_size,
                    in_channels=14,
                    global_features=20,
                    history_length=history_length,
                    policy_size=policy_size,
                    num_players=model_num_players,
                    num_res_blocks=effective_blocks,
                    num_filters=effective_filters,
                )

                if self.is_main_process:
                    logger.info(
                        f"Initializing RingRiftCNN_v3 (spatial): board_size={board_size}, "
                        f"policy_size={policy_size}, num_players={model_num_players}"
                    )

                return model

        else:  # v2 default
            from app.ai.neural_net.cnn_v2 import RingRiftCNN_v2

            return RingRiftCNN_v2(
                board_size=board_size,
                in_channels=14,
                global_features=20,
                history_length=history_length,
                policy_size=policy_size,
                num_res_blocks=effective_blocks,
                num_filters=effective_filters,
                num_players=model_num_players,
            )

    def _validate_value_head(
        self,
        model: nn.Module,
        context: str = "",
    ) -> None:
        """Validate model value head matches expected players.

        Args:
            model: Model to validate
            context: Context string for error messages

        Raises:
            ValueError: If value head doesn't match expected players
        """
        expected_players = self.config.num_players
        ctx = f" ({context})" if context else ""

        # Check model's num_players attribute
        if hasattr(model, "num_players"):
            model_players = model.num_players
            if model_players != expected_players:
                raise ValueError(
                    f"Model value head mismatch{ctx}: model.num_players={model_players} "
                    f"but training expects {expected_players} players. "
                    f"Use transfer_2p_to_4p.py to resize value head."
                )

        # Check final value layer output
        final_layer = None
        if hasattr(model, "value_fc3"):
            final_layer = model.value_fc3
        elif hasattr(model, "value_fc2"):
            final_layer = model.value_fc2

        if final_layer is not None:
            out_features = final_layer.out_features
            if out_features != expected_players:
                layer_name = "value_fc3" if hasattr(model, "value_fc3") else "value_fc2"
                raise ValueError(
                    f"{layer_name} output mismatch{ctx}: out_features={out_features} "
                    f"but training expects {expected_players} players. "
                    f"Use transfer_2p_to_4p.py to resize value head."
                )

    def _apply_freeze_policy(self, model: nn.Module) -> None:
        """Freeze all parameters except value head.

        Args:
            model: Model to apply freeze to
        """
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze value head parameters
        value_head_params = []
        for name, param in model.named_parameters():
            if any(
                x in name.lower()
                for x in ["value_fc", "value_head", "value_conv", "value_bn"]
            ):
                param.requires_grad = True
                value_head_params.append(name)
                if self.is_main_process:
                    logger.info(f"[freeze_policy] Unfreezing: {name}")

        if not value_head_params and self.is_main_process:
            logger.warning(
                "[freeze_policy] No value head parameters found to unfreeze!"
            )


# =============================================================================
# Factory functions
# =============================================================================


def create_model_initializer(
    board_type: "BoardType",
    num_players: int,
    device: torch.device,
    model_version: str = "v2",
    **kwargs: Any,
) -> ModelInitializer:
    """Create a ModelInitializer with the specified settings.

    Args:
        board_type: Board type
        num_players: Number of players
        device: Target device
        model_version: Model version string
        **kwargs: Additional ModelConfig parameters

    Returns:
        Configured ModelInitializer
    """
    config = ModelConfig(
        board_type=board_type,
        num_players=num_players,
        model_version=model_version,
        **{k: v for k, v in kwargs.items() if hasattr(ModelConfig, k)},
    )
    return ModelInitializer(config, device, **{
        k: v for k, v in kwargs.items()
        if k in ("distributed", "is_main_process")
    })
