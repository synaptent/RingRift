"""Model factory for RingRift neural network training.

This module centralizes model creation logic, extracting what was previously
~200 lines of model initialization code from train_model().

December 2025: Extracted from train.py to improve modularity.
"""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from app.models import BoardType
from app.ai.neural_net import (
    HEX8_BOARD_SIZE,
    HEX_BOARD_SIZE,
    MAX_PLAYERS,
    HexNeuralNet_v2,
    HexNeuralNet_v3,
    HexNeuralNet_v3_Flat,
    HexNeuralNet_v4,
    HexNeuralNet_v5_Heavy,
    RingRiftCNN_v2,
    RingRiftCNN_v3,
    RingRiftCNN_v3_Flat,
    get_policy_size_for_board,
)
from app.training.checkpoint_unified import load_checkpoint
from app.training.schedulers import create_lr_scheduler
from app.training.train_model_factory import create_training_model
from app.training.training_enhancements import EarlyStopping
from app.utils.torch_utils import safe_load_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model creation."""

    board_type: BoardType
    board_size: int
    policy_size: int
    in_channels: int
    global_features: int = 20
    history_length: int = 3
    num_players: int = 2
    multi_player: bool = False
    model_version: str = 'v2'
    num_res_blocks: int = 6
    num_filters: int = 96
    dropout: float = 0.08
    feature_version: int = 1


@dataclass
class TrainingModelArtifacts:
    """Model and optimizer state prepared for training."""

    model: nn.Module
    optimizer: optim.Optimizer
    epoch_scheduler: Any
    plateau_scheduler: Any
    eval_feedback_handler: Any
    early_stopper: Any
    start_epoch: int


def create_model(config: ModelConfig, device: torch.device | None = None) -> nn.Module:
    """Create a neural network model based on configuration.

    Args:
        config: Model configuration
        device: Device to place model on (optional)

    Returns:
        Initialized neural network model
    """
    use_hex_model = config.board_type in (BoardType.HEXAGONAL, BoardType.HEX8)
    use_hex_v3 = use_hex_model and config.model_version == 'v3'
    # Compute hex_radius from board_type: HEX8 has radius 4, HEXAGONAL has radius 12
    hex_radius = 4 if config.board_type == BoardType.HEX8 else 12

    # Determine effective number of players for value head
    if config.multi_player:
        effective_num_players = MAX_PLAYERS
    else:
        effective_num_players = config.num_players

    if use_hex_v3:
        # HexNeuralNet_v3 for hexagonal boards with spatial policy heads
        model = HexNeuralNet_v3(
            in_channels=config.in_channels,
            global_features=config.global_features,
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters,
            board_size=config.board_size,
            hex_radius=hex_radius,
            policy_size=config.policy_size,
            num_players=effective_num_players,
        )
        logger.info(
            f"Created HexNeuralNet_v3: board_size={config.board_size}, "
            f"hex_radius={hex_radius}, policy_size={config.policy_size}, "
            f"in_channels={config.in_channels}"
        )
    elif use_hex_model:
        # HexNeuralNet_v2 for hexagonal boards
        model = HexNeuralNet_v2(
            in_channels=config.in_channels,
            global_features=config.global_features,
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters,
            board_size=config.board_size,
            hex_radius=hex_radius,
            policy_size=config.policy_size,
            num_players=effective_num_players,
        )
        logger.info(
            f"Created HexNeuralNet_v2: board_size={config.board_size}, "
            f"hex_radius={hex_radius}, policy_size={config.policy_size}, "
            f"in_channels={config.in_channels}"
        )
    elif config.model_version == 'v4':
        # V4 NAS-optimized architecture
        from app.ai.neural_net import RingRiftCNN_v4
        model = RingRiftCNN_v4(
            board_size=config.board_size,
            in_channels=14,  # 14 spatial feature channels per frame
            global_features=config.global_features,
            history_length=config.history_length,
            policy_size=config.policy_size,
            num_players=effective_num_players,
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters,
            num_attention_heads=4,  # NAS optimal
            dropout=config.dropout,
            initial_kernel_size=5,  # NAS optimal
        )
        logger.info(
            f"Created RingRiftCNN_v4 (NAS): board_size={config.board_size}, "
            f"policy_size={config.policy_size}, blocks={config.num_res_blocks}, "
            f"filters={config.num_filters}"
        )
    elif config.model_version == 'v3':
        # V3 architecture with spatial policy heads
        model = RingRiftCNN_v3(
            board_size=config.board_size,
            in_channels=14,
            global_features=config.global_features,
            history_length=config.history_length,
            policy_size=config.policy_size,
            num_players=effective_num_players,
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters,
        )
        logger.info(
            f"Created RingRiftCNN_v3: board_size={config.board_size}, "
            f"policy_size={config.policy_size}, num_players={effective_num_players}"
        )
    else:
        # RingRiftCNN_v2 for square boards (default)
        model = RingRiftCNN_v2(
            board_size=config.board_size,
            in_channels=14,
            global_features=config.global_features,
            history_length=config.history_length,
            policy_size=config.policy_size,
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters,
            num_players=effective_num_players if config.multi_player else 2,
        )
        logger.info(
            f"Created RingRiftCNN_v2: board_size={config.board_size}, "
            f"policy_size={config.policy_size}"
        )

    # Set feature version for compatibility checking
    with contextlib.suppress(Exception):
        model.feature_version = config.feature_version

    # Move to device if specified
    if device is not None:
        model.to(device)

    return model


def get_board_size(board_type: BoardType) -> int:
    """Get the canonical board size for a board type."""
    if board_type == BoardType.SQUARE19:
        return 19
    elif board_type == BoardType.HEXAGONAL:
        return HEX_BOARD_SIZE  # 25
    elif board_type == BoardType.HEX8:
        return HEX8_BOARD_SIZE  # 9
    else:
        return 8  # Default square8


def compute_in_channels(
    board_type: BoardType,
    history_length: int,
    model_version: str = 'v2',
) -> int:
    """Compute the number of input channels based on board type and history.

    Args:
        board_type: The board type
        history_length: Number of history frames
        model_version: Model version (affects hex channel count)

    Returns:
        Number of input channels for the model
    """
    use_hex = board_type in (BoardType.HEXAGONAL, BoardType.HEX8)
    use_hex_v3 = use_hex and model_version == 'v3'

    if use_hex_v3:
        base_channels = 16
    elif use_hex:
        base_channels = 10
    else:
        base_channels = 14

    return base_channels * (history_length + 1)


def get_effective_architecture(
    model_version: str,
    board_type: BoardType,
    num_res_blocks: int | None = None,
    num_filters: int | None = None,
) -> tuple[int, int]:
    """Get effective architecture parameters.

    Args:
        model_version: Model version (v2, v3, v4)
        board_type: Board type (affects defaults for hex)
        num_res_blocks: Override for residual blocks
        num_filters: Override for filters

    Returns:
        Tuple of (effective_blocks, effective_filters)
    """
    use_hex = board_type in (BoardType.HEXAGONAL, BoardType.HEX8)

    if model_version == 'v3' or use_hex:
        default_blocks = 12
        default_filters = 192
    elif model_version == 'v4':
        default_blocks = 13
        default_filters = 128
    else:
        default_blocks = 6
        default_filters = 96

    effective_blocks = num_res_blocks if num_res_blocks is not None else default_blocks
    effective_filters = num_filters if num_filters is not None else default_filters

    return effective_blocks, effective_filters


def load_model_weights(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
) -> bool:
    """Load model weights from a checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load onto
        strict: Whether to require exact key match

    Returns:
        True if loaded successfully, False otherwise
    """
    import os

    from app.utils.torch_utils import safe_load_checkpoint

    if not os.path.exists(checkpoint_path):
        return False

    try:
        checkpoint = safe_load_checkpoint(
            checkpoint_path,
            map_location=device,
            warn_on_unsafe=False,
        )

        # Handle both raw state_dict and checkpoint dict formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=strict)
        logger.info(f"Loaded model weights from {checkpoint_path}")
        return True

    except Exception as e:
        logger.warning(f"Could not load weights from {checkpoint_path}: {e}")
        return False


def wrap_model_ddp(
    model: nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """Wrap model with DistributedDataParallel.

    Args:
        model: Model to wrap
        device: Device model is on
        find_unused_parameters: Whether to find unused parameters

    Returns:
        DDP-wrapped model
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    if device.type == 'cuda':
        device_ids = [device.index if device.index is not None else 0]
    else:
        device_ids = None

    return DDP(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_model_summary(model: nn.Module, config: ModelConfig) -> None:
    """Log a summary of the model architecture."""
    param_count = count_parameters(model)
    logger.info(
        f"Model summary: {param_count:,} trainable parameters, "
        f"board_size={config.board_size}, policy_size={config.policy_size}"
    )


def validate_model_value_head(
    model: nn.Module,
    expected_players: int,
    context: str = "",
) -> None:
    """Validate that the model value head matches the expected player count."""
    ctx = f" ({context})" if context else ""

    if hasattr(model, "num_players"):
        model_players = model.num_players
        if model_players != expected_players:
            raise ValueError(
                f"Model value head mismatch{ctx}: model.num_players={model_players} "
                f"but training expects {expected_players} players. "
                "Use transfer_2p_to_4p.py to resize value head."
            )

    final_value_layer = None
    if hasattr(model, "value_fc3"):
        final_value_layer = model.value_fc3
    elif hasattr(model, "value_fc2"):
        final_value_layer = model.value_fc2

    if final_value_layer is not None:
        out_features = final_value_layer.out_features
        if out_features != expected_players:
            layer_name = "value_fc3" if hasattr(model, "value_fc3") else "value_fc2"
            raise ValueError(
                f"{layer_name} output mismatch{ctx}: out_features={out_features} "
                f"but training expects {expected_players} players. "
                "Use transfer_2p_to_4p.py to resize value head."
            )

    if hasattr(model, "value_head"):
        value_head = model.value_head
        if hasattr(value_head, "out_features"):
            out_features = value_head.out_features
            if out_features != expected_players:
                raise ValueError(
                    f"value_head output mismatch{ctx}: out_features={out_features} "
                    f"but training expects {expected_players} players."
                )
        elif isinstance(value_head, nn.Sequential):
            last_layer = list(value_head.modules())[-1]
            if hasattr(last_layer, "out_features"):
                out_features = last_layer.out_features
                if out_features != expected_players:
                    raise ValueError(
                        f"value_head output mismatch{ctx}: last layer out_features={out_features} "
                        f"but training expects {expected_players} players."
                    )


def prepare_training_model_artifacts(
    *,
    config: Any,
    model_version: str,
    model_type: str,
    board_size: int,
    policy_size: int,
    num_players: int,
    encoding_channels: int,
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
    distributed: bool,
    is_main: bool,
    device: torch.device,
    enhancements_manager: Any,
    gradient_checkpointing: bool,
    auto_tune_batch_size: bool,
    target_memory_fraction: float | None,
    safe_mode: bool,
    save_path: str,
    init_weights_path: str | None,
    init_weights_strict: bool,
    resume_path: str | None,
    find_unused_parameters: bool,
    warmup_epochs: int,
    lr_scheduler: str | None,
    lr_min: float | None,
    lr_t0: int,
    lr_t_mult: int,
    freeze_policy: bool,
    early_stopping_patience: int,
    elo_early_stopping_patience: int,
    elo_min_improvement: float | None,
    checkpoint_dir: str,
    data_path_str: str,
    has_training_enhancements: bool,
    evaluation_feedback_handler_cls: Any,
) -> TrainingModelArtifacts:
    """Build the model and optimizer/scheduler state for a training run."""
    model = create_training_model(
        config=config,
        model_version=model_version,
        model_type=model_type,
        board_size=board_size,
        policy_size=policy_size,
        num_players=num_players,
        encoding_channels=encoding_channels,
        hex_radius=hex_radius,
        hex_num_players=hex_num_players,
        use_hex_model=use_hex_model,
        use_hex_v3=use_hex_v3,
        use_hex_v4=use_hex_v4,
        use_hex_v5=use_hex_v5,
        use_hex_v5_large=use_hex_v5_large,
        detected_num_heuristics=detected_num_heuristics,
        effective_blocks=effective_blocks,
        effective_filters=effective_filters,
        multi_player=multi_player,
        dropout=dropout,
        config_feature_version=config_feature_version,
        distributed=distributed,
        is_main=is_main,
        HexNeuralNet_v2=HexNeuralNet_v2,
        HexNeuralNet_v3=HexNeuralNet_v3,
        HexNeuralNet_v3_Flat=HexNeuralNet_v3_Flat,
        HexNeuralNet_v4=HexNeuralNet_v4,
        HexNeuralNet_v5_Heavy=HexNeuralNet_v5_Heavy,
        RingRiftCNN_v2=RingRiftCNN_v2,
        RingRiftCNN_v3=RingRiftCNN_v3,
        RingRiftCNN_v3_Flat=RingRiftCNN_v3_Flat,
        MAX_PLAYERS=MAX_PLAYERS,
    )
    model.to(device)

    try:
        model_channels = None
        if hasattr(model, "conv1") and hasattr(model.conv1, "weight"):
            model_channels = model.conv1.weight.shape[1]
        elif hasattr(model, "in_channels"):
            model_channels = model.in_channels
        if model_channels is not None and encoding_channels is not None:
            if model_channels != encoding_channels:
                raise ValueError(
                    f"ENCODING MISMATCH: NPZ has {encoding_channels} feature channels "
                    f"but model expects {model_channels} channels. "
                    "This will produce a garbage model. "
                    "Check encoder version (v2=40ch, v3/v4=64ch, v5=56ch)."
                )
            logger.info(
                "[EncodingContract] Verified: NPZ channels (%s) match model channels (%s)",
                encoding_channels,
                model_channels,
            )
    except ValueError:
        raise
    except (AttributeError, IndexError, TypeError) as exc:
        logger.debug("[EncodingContract] Could not verify channels: %s", exc)

    if gradient_checkpointing:
        try:
            from app.training.gradient_checkpointing import GradientCheckpointing

            gc_manager = GradientCheckpointing(model)
            gc_manager.enable()
            if is_main:
                logger.info("[GradientCheckpointing] Enabled - trading compute for memory")
        except ImportError as exc:
            logger.warning("[GradientCheckpointing] Failed to enable: %s", exc)

    validate_model_value_head(model, num_players, "after model creation")

    if enhancements_manager is not None:
        enhancements_manager.model = model
        enhancements_manager.initialize_all()

    if auto_tune_batch_size and str(device).startswith("cuda"):
        try:
            from app.training.config import (
                get_gpu_scaling_config,
                get_optimal_batch_size_from_gpu_memory,
            )

            original_batch = config.batch_size
            model_params = sum(p.numel() for p in model.parameters())
            feature_channels = getattr(model, "in_channels", 56)
            gpu_config = get_gpu_scaling_config()
            effective_memory_fraction = target_memory_fraction
            if effective_memory_fraction is None and safe_mode:
                effective_memory_fraction = gpu_config.safe_mode_memory_fraction

            mode_str = "[SAFE MODE]" if safe_mode else ""
            logger.info(
                "[AutoBatchSize]%s Calculating optimal batch size from GPU memory...",
                mode_str,
            )
            logger.info(
                "[AutoBatchSize] Model params: %s, board_size: %s, num_players: %s",
                f"{model_params:,}",
                board_size,
                num_players,
            )
            if effective_memory_fraction:
                logger.info(
                    "[AutoBatchSize] Memory target: %.0f%%",
                    effective_memory_fraction * 100,
                )

            config.batch_size = get_optimal_batch_size_from_gpu_memory(
                model_params=model_params,
                feature_channels=feature_channels,
                board_size=board_size,
                num_players=num_players,
                target_memory_fraction=effective_memory_fraction,
                min_batch=64,
                max_batch=4096,
                config=gpu_config,
            )
            logger.info(
                "[AutoBatchSize] Auto-tuned batch size: %s (was %s)",
                config.batch_size,
                original_batch,
            )
        except (RuntimeError, ValueError, ImportError) as exc:
            logger.warning(
                "[AutoBatchSize] Batch size auto-tuning failed: %s. Using original batch size.",
                exc,
            )

    if init_weights_path is None and not os.path.exists(save_path):
        board_type_str = (
            config.board_type.value if hasattr(config.board_type, "value") else str(config.board_type)
        )
        canonical_path = f"models/canonical_{board_type_str}_{num_players}p.pth"
        if os.path.exists(canonical_path):
            try:
                from app.ai.neural_net.architecture_registry import (
                    get_encoder_version_from_checkpoint,
                )

                canonical_encoder = get_encoder_version_from_checkpoint(canonical_path)
                data_encoder = None
                if encoding_channels == 40:
                    data_encoder = "v2"
                elif encoding_channels == 64:
                    data_encoder = "v3"
                elif encoding_channels == 56:
                    data_encoder = "v2"

                if canonical_encoder and data_encoder and canonical_encoder == data_encoder:
                    init_weights_path = canonical_path
                    if is_main:
                        logger.info(
                            "[AutoInitWeights] Using canonical model as starting point: %s",
                            canonical_path,
                        )
                elif is_main:
                    logger.info(
                        "[AutoInitWeights] Canonical model %s has encoder %s, but data uses %s. Training from scratch instead.",
                        canonical_path,
                        canonical_encoder,
                        data_encoder,
                    )
            except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                if is_main:
                    logger.warning(
                        "[AutoInitWeights] Could not check canonical model compatibility: %s. Training from scratch.",
                        exc,
                    )
        elif is_main:
            logger.info(
                "[AutoInitWeights] No canonical model found at %s, training from scratch",
                canonical_path,
            )

    if init_weights_path is not None and os.path.exists(init_weights_path):
        try:
            from app.ai.neural_net.architecture_registry import (
                get_encoder_version_from_checkpoint,
                get_model_version_from_checkpoint,
            )

            init_encoder_version = get_encoder_version_from_checkpoint(init_weights_path)
            init_model_version = get_model_version_from_checkpoint(init_weights_path)
            data_encoder = None
            if encoding_channels == 40:
                data_encoder = "v2"
            elif encoding_channels == 64:
                data_encoder = "v3"
            elif encoding_channels == 56:
                data_encoder = "v2"

            if init_encoder_version and data_encoder and init_encoder_version != data_encoder:
                error_msg = (
                    f"\n{'=' * 70}\n"
                    "ENCODER MISMATCH DETECTED (FAIL-FAST)\n"
                    f"{'=' * 70}\n\n"
                    f"Init weights: {init_weights_path}\n"
                    f"  - Encoder: {init_encoder_version} ({40 if init_encoder_version == 'v2' else 64} channels)\n\n"
                    f"Training data: {data_path_str}\n"
                    f"  - Encoder: {data_encoder} ({encoding_channels} channels)\n\n"
                    f"PROBLEM: Cannot train {data_encoder} data with {init_encoder_version} model weights.\n\n"
                    "SOLUTIONS:\n"
                    f"  1. Re-export training data with --encoder-version {init_encoder_version}\n"
                    f"  2. Use a different init_weights file matching {data_encoder}\n"
                    "  3. Train from scratch without --init-weights\n"
                    f"{'=' * 70}"
                )
                if is_main:
                    logger.error(error_msg)
                raise ValueError(
                    f"Encoder mismatch: init_weights={init_encoder_version}, data={data_encoder}"
                )

            if init_model_version:
                if model_version != init_model_version:
                    if is_main:
                        logger.warning(
                            "[ArchValidation] Model version mismatch detected!\n"
                            "  Init weights uses: %s\n"
                            "  Training configured for: %s\n"
                            "  Auto-adapting to use: %s",
                            init_model_version,
                            model_version,
                            init_model_version,
                        )
                    model_version = init_model_version
                elif is_main:
                    logger.info(
                        "[ArchValidation] Architecture validated: encoder=%s, model=%s",
                        init_encoder_version,
                        init_model_version,
                    )
        except ImportError:
            pass
        except FileNotFoundError:
            pass

    if init_weights_path is not None and os.path.exists(init_weights_path):
        try:
            from app.training.checkpointing import load_weights_only

            load_result = load_weights_only(
                init_weights_path,
                model,
                device=device,
                strict=init_weights_strict,
            )
            if is_main:
                logger.info("Loaded initial weights from %s", init_weights_path)
                if load_result.get("missing_keys"):
                    logger.info(
                        "  Missing keys (will be randomly initialized): %d",
                        len(load_result["missing_keys"]),
                    )
                if load_result.get("unexpected_keys"):
                    logger.info(
                        "  Unexpected keys (ignored): %d",
                        len(load_result["unexpected_keys"]),
                    )
            validate_model_value_head(model, num_players, "after loading init_weights")
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            if is_main:
                logger.warning(
                    "Could not load init weights from %s: %s. Starting fresh.",
                    init_weights_path,
                    exc,
                )

    if os.path.exists(save_path) and init_weights_path is None:
        try:
            checkpoint = safe_load_checkpoint(
                save_path,
                map_location=device,
                warn_on_unsafe=False,
            )
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            if is_main:
                logger.info("Loaded existing model weights from %s", save_path)
            validate_model_value_head(model, num_players, "after loading checkpoint")
        except (OSError, RuntimeError, ValueError, KeyError):
            pass
    elif os.path.exists(save_path) and init_weights_path is not None and is_main:
        logger.info(
            "Skipping save_path resume (%s) - init_weights takes priority (AlphaZero pattern)",
            save_path,
        )

    if distributed:
        model = wrap_model_ddp(
            model,
            device,
            find_unused_parameters=find_unused_parameters,
        )
        if is_main:
            logger.info("Model wrapped with DistributedDataParallel")

    if freeze_policy:
        for param in model.parameters():
            param.requires_grad = False
        value_head_params = []
        for name, param in model.named_parameters():
            if any(token in name.lower() for token in ["value_fc", "value_head", "value_conv", "value_bn"]):
                param.requires_grad = True
                value_head_params.append(param)
                logger.info("[freeze_policy] Unfreezing: %s", name)
        if not value_head_params:
            logger.warning(
                "[freeze_policy] No value head parameters found! Check model architecture. Training all parameters."
            )
            for param in model.parameters():
                param.requires_grad = True
            optimizer_params = model.parameters()
        else:
            logger.info(
                "[freeze_policy] Training only %d value head parameters",
                len(value_head_params),
            )
            optimizer_params = value_head_params
    else:
        optimizer_params = model.parameters()

    optimizer = optim.Adam(
        optimizer_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    epoch_scheduler = create_lr_scheduler(
        optimizer,
        scheduler_type=lr_scheduler,
        total_epochs=config.epochs_per_iter,
        warmup_epochs=warmup_epochs,
        lr_min=lr_min,
        lr_t0=lr_t0,
        lr_t_mult=lr_t_mult,
    )
    plateau_scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )
        if epoch_scheduler is None
        else None
    )

    eval_feedback_handler = None
    if has_training_enhancements and evaluation_feedback_handler_cls is not None:
        config_key = f"{config.board_type.value}_{num_players}p"
        eval_feedback_handler = evaluation_feedback_handler_cls(
            optimizer=optimizer,
            config_key=config_key,
            min_lr=lr_min or 1e-6,
            max_lr=config.learning_rate * 2,
        )
        if eval_feedback_handler.subscribe():
            if is_main:
                logger.info(
                    "[EvaluationFeedbackHandler] Enabled for %s (LR adjusted based on Elo trends)",
                    config_key,
                )
        else:
            eval_feedback_handler = None

    early_stopper = None
    if early_stopping_patience > 0 or elo_early_stopping_patience > 0:
        early_stopper = EarlyStopping(
            patience=early_stopping_patience if early_stopping_patience > 0 else 999999,
            min_delta=0.0001,
            elo_patience=elo_early_stopping_patience if elo_early_stopping_patience > 0 else None,
            elo_min_improvement=elo_min_improvement,
            config_name=f"{config.board_type.value}_{num_players}p",
        )

    start_epoch = 0
    if resume_path is not None and os.path.exists(resume_path):
        model_to_load = model.module if distributed else model
        start_epoch, _ = load_checkpoint(
            resume_path,
            model_to_load,
            optimizer,
            scheduler=epoch_scheduler,
            early_stopping=early_stopper,
            device=device,
        )
        start_epoch += 1
        if is_main:
            logger.info("Resuming training from epoch %s", start_epoch)

    os.makedirs(checkpoint_dir, exist_ok=True)

    return TrainingModelArtifacts(
        model=model,
        optimizer=optimizer,
        epoch_scheduler=epoch_scheduler,
        plateau_scheduler=plateau_scheduler,
        eval_feedback_handler=eval_feedback_handler,
        early_stopper=early_stopper,
        start_epoch=start_epoch,
    )


__all__ = [
    'ModelConfig',
    'TrainingModelArtifacts',
    'compute_in_channels',
    'count_parameters',
    'create_model',
    'get_board_size',
    'get_effective_architecture',
    'load_model_weights',
    'log_model_summary',
    'prepare_training_model_artifacts',
    'validate_model_value_head',
    'wrap_model_ddp',
]
