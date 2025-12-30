"""Bridge module for train_model() component integration.

December 2025: Provides helpers to incrementally adopt the new modular
components without rewriting train_model() entirely.

This module provides:
1. resolve_train_config() - Uses TrainConfigResolver
2. validate_training_data() - Uses DataValidator
3. initialize_training_model() - Uses ModelInitializer
4. build_train_context() - Builds TrainContext from parameters

Usage:
    from app.training.train_components import (
        resolve_train_config,
        validate_training_data,
        initialize_training_model,
        build_train_context,
    )

    # In train_model():
    resolved = resolve_train_config(config, **kwargs)
    validation = validate_training_data(data_paths, board_type, num_players, resolved)
    model_result = initialize_training_model(config, resolved, data_path)
    context = build_train_context(config, resolved, model_result, ...)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from app.training.train_context import (
    TrainContext,
    ResolvedConfig,
    TrainingProgress,
)
from app.training.train_config_resolver import (
    TrainConfigResolver,
    resolve_device,
)
from app.training.data_validator import (
    DataValidator,
    DataValidationConfig,
    DataValidationResult,
)
from app.training.model_initializer import (
    ModelInitializer,
    ModelConfig,
    ModelInitResult,
)

if TYPE_CHECKING:
    from app.training.train_config import TrainConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Resolution
# =============================================================================


def resolve_train_config(
    config: "TrainConfig",
    early_stopping_patience: int | None = None,
    elo_early_stopping_patience: int | None = None,
    elo_min_improvement: float | None = None,
    warmup_epochs: int | None = None,
    lr_scheduler: str | None = None,
    lr_min: float | None = None,
    lr_t0: int = 10,
    lr_t_mult: int = 2,
    checkpoint_dir: str = "checkpoints",
    checkpoint_interval: int = 5,
    distributed: bool = False,
    local_rank: int = -1,
    num_players: int = 2,
    **kwargs: Any,
) -> ResolvedConfig:
    """Resolve training configuration using TrainConfigResolver.

    This replaces the parameter resolution logic at the start of train_model().

    Args:
        config: Base training configuration
        early_stopping_patience: Early stopping patience (None = use config/default)
        elo_early_stopping_patience: Elo-based early stopping patience
        elo_min_improvement: Minimum Elo improvement for patience reset
        warmup_epochs: LR warmup epochs
        lr_scheduler: LR scheduler type
        lr_min: Minimum learning rate
        lr_t0: CosineAnnealingWarmRestarts T_0
        lr_t_mult: CosineAnnealingWarmRestarts T_mult
        checkpoint_dir: Checkpoint directory
        checkpoint_interval: Checkpoint save interval
        distributed: Distributed training mode
        local_rank: Local rank for distributed
        num_players: Number of players
        **kwargs: Additional override parameters

    Returns:
        Fully resolved configuration
    """
    resolver = TrainConfigResolver(config)

    # Build overrides dictionary
    overrides = {
        "early_stopping_patience": early_stopping_patience,
        "elo_early_stopping_patience": elo_early_stopping_patience,
        "elo_min_improvement": elo_min_improvement,
        "warmup_epochs": warmup_epochs,
        "lr_scheduler": lr_scheduler,
        "lr_min": lr_min,
        "lr_t0": lr_t0,
        "lr_t_mult": lr_t_mult,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_interval": checkpoint_interval,
        "distributed": distributed,
        "local_rank": local_rank,
        "num_players": num_players,
    }

    # Add any extra kwargs
    overrides.update(kwargs)

    # Resolve with precedence: override > config > default
    resolved = resolver.resolve(**overrides)

    return resolved


def resolve_training_device(
    distributed: bool = False,
    local_rank: int = -1,
) -> torch.device:
    """Resolve the training device.

    Args:
        distributed: Distributed training mode
        local_rank: Local rank for distributed

    Returns:
        Selected torch device
    """
    return resolve_device(distributed=distributed, local_rank=local_rank)


# =============================================================================
# Data Validation
# =============================================================================


def validate_training_data(
    data_paths: list[str],
    board_type: str,
    num_players: int,
    resolved: ResolvedConfig,
    is_main_process: bool = True,
) -> DataValidationResult:
    """Validate training data using DataValidator.

    This replaces the freshness check and NPZ validation logic in train_model().

    Args:
        data_paths: Paths to training data files
        board_type: Board type string (e.g., "square8", "hex8")
        num_players: Number of players
        resolved: Resolved training configuration
        is_main_process: Whether this is the main process

    Returns:
        Validation result with all check outcomes
    """
    validator_config = DataValidationConfig.from_resolved(resolved)
    validator = DataValidator(validator_config)

    result = validator.validate_all(
        data_paths=data_paths,
        board_type=board_type,
        num_players=num_players,
        distributed=resolved.distributed,
        is_main_process=is_main_process,
    )

    return result


# =============================================================================
# Model Initialization
# =============================================================================


def initialize_training_model(
    config: "TrainConfig",
    resolved: ResolvedConfig,
    data_path: str | list[str],
    device: torch.device | None = None,
    is_main_process: bool = True,
) -> ModelInitResult:
    """Initialize training model using ModelInitializer.

    This replaces the model creation and weight loading logic in train_model().

    Args:
        config: Training configuration
        resolved: Resolved configuration
        data_path: Path(s) to training data
        device: Device to use (None = auto-detect)
        is_main_process: Whether this is the main process

    Returns:
        ModelInitResult with model, metadata, and any errors
    """
    # Resolve device if not provided
    if device is None:
        device = resolve_training_device(
            distributed=resolved.distributed,
            local_rank=resolved.local_rank,
        )

    # Create model configuration from resolved settings
    model_config = ModelConfig(
        board_type=config.board_type,
        num_players=resolved.num_players,
        multi_player=resolved.multi_player,
        model_version=resolved.model_version,
        model_type=resolved.model_type,
        num_res_blocks=resolved.num_res_blocks,
        num_filters=resolved.num_filters,
        history_length=getattr(config, "history_length", 3),
        dropout=resolved.dropout,
        freeze_policy=resolved.freeze_policy,
        feature_version=getattr(resolved, "feature_version", 1),
        model_id=getattr(config, "model_id", ""),
    )

    initializer = ModelInitializer(
        config=model_config,
        device=device,
        distributed=resolved.distributed,
        is_main_process=is_main_process,
    )

    # Get first data path for metadata extraction
    if isinstance(data_path, list):
        primary_data_path = data_path[0] if data_path else None
    else:
        primary_data_path = data_path

    result = initializer.create_model(data_path=primary_data_path)

    return result


# =============================================================================
# Context Building
# =============================================================================


def build_train_context(
    config: "TrainConfig",
    resolved: ResolvedConfig,
    model_result: ModelInitResult,
    data_paths: list[str],
    save_path: str,
    device: torch.device,
    optimizer: Any = None,
    train_loader: Any = None,
    val_loader: Any = None,
    **components: Any,
) -> TrainContext:
    """Build a TrainContext from all training components.

    This consolidates all training state into a single context object.

    Args:
        config: Training configuration
        resolved: Resolved configuration
        model_result: Result from model initialization
        data_paths: Training data paths
        save_path: Model save path
        device: Training device
        optimizer: Optimizer (optional, can be set later)
        train_loader: Training data loader (optional)
        val_loader: Validation data loader (optional)
        **components: Additional components (schedulers, handlers, etc.)

    Returns:
        Configured TrainContext
    """
    context = TrainContext(
        config=config,
        resolved=resolved,
        data_paths=data_paths,
        save_path=save_path,
        device=device,
        distributed=resolved.distributed,
        local_rank=resolved.local_rank,
        world_size=components.get("world_size", 1),
        model=model_result.model,
        model_version=model_result.model_version,
        policy_size=model_result.policy_size,
        board_size=model_result.board_size,
        effective_blocks=model_result.effective_blocks,
        effective_filters=model_result.effective_filters,
        feature_version=model_result.feature_version,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config_label=f"{config.board_type.value}_{resolved.num_players}p",
    )

    # Set optional components
    if "epoch_scheduler" in components:
        context.epoch_scheduler = components["epoch_scheduler"]
    if "plateau_scheduler" in components:
        context.plateau_scheduler = components["plateau_scheduler"]
    if "grad_scaler" in components:
        context.grad_scaler = components["grad_scaler"]
    if "train_sampler" in components:
        context.train_sampler = components["train_sampler"]
    if "early_stopper" in components:
        context.early_stopper = components["early_stopper"]
    if "enhancements_manager" in components:
        context.enhancements_manager = components["enhancements_manager"]
    if "training_facade" in components:
        context.training_facade = components["training_facade"]
    if "hard_example_miner" in components:
        context.hard_example_miner = components["hard_example_miner"]
    if "quality_trainer" in components:
        context.quality_trainer = components["quality_trainer"]
    if "hot_buffer" in components:
        context.hot_buffer = components["hot_buffer"]
    if "training_breaker" in components:
        context.training_breaker = components["training_breaker"]
    if "anomaly_detector" in components:
        context.anomaly_detector = components["anomaly_detector"]
    if "adaptive_clipper" in components:
        context.adaptive_clipper = components["adaptive_clipper"]
    if "shutdown_handler" in components:
        context.shutdown_handler = components["shutdown_handler"]
    if "checkpoint_averager" in components:
        context.checkpoint_averager = components["checkpoint_averager"]
    if "async_checkpointer" in components:
        context.async_checkpointer = components["async_checkpointer"]
    if "eval_feedback_handler" in components:
        context.eval_feedback_handler = components["eval_feedback_handler"]
    if "calibration_tracker" in components:
        context.calibration_tracker = components["calibration_tracker"]
    if "metrics_collector" in components:
        context.metrics_collector = components["metrics_collector"]
    if "heartbeat_monitor" in components:
        context.heartbeat_monitor = components["heartbeat_monitor"]
    if "gradient_surgeon" in components:
        context.gradient_surgeon = components["gradient_surgeon"]

    return context


# =============================================================================
# Utility Functions
# =============================================================================


def get_data_paths(data_path: str | list[str]) -> list[str]:
    """Normalize data path to list.

    Args:
        data_path: Single path or list of paths

    Returns:
        List of paths
    """
    if isinstance(data_path, str):
        return [data_path]
    return list(data_path)


def check_is_main_process(distributed: bool, local_rank: int = -1) -> bool:
    """Check if this is the main process in distributed training.

    Args:
        distributed: Distributed mode flag
        local_rank: Local rank

    Returns:
        True if main process
    """
    if not distributed:
        return True
    return local_rank <= 0


# =============================================================================
# Integration Example
# =============================================================================


def train_model_with_components(
    config: "TrainConfig",
    data_path: str | list[str],
    save_path: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Example of train_model using the new components.

    This demonstrates how to gradually adopt the new architecture.
    Not intended for production use yet - shows the pattern.

    Args:
        config: Training configuration
        data_path: Path(s) to training data
        save_path: Model save path
        **kwargs: Training parameters

    Returns:
        Training result dictionary
    """
    # Phase 1: Resolve configuration
    resolved = resolve_train_config(config, **kwargs)

    # Phase 2: Resolve device
    device = resolve_training_device(
        distributed=resolved.distributed,
        local_rank=resolved.local_rank,
    )
    is_main = check_is_main_process(resolved.distributed, resolved.local_rank)

    # Phase 3: Validate data
    data_paths = get_data_paths(data_path)
    board_type = config.board_type.value if hasattr(config.board_type, "value") else str(config.board_type)

    validation = validate_training_data(
        data_paths=data_paths,
        board_type=board_type,
        num_players=resolved.num_players,
        resolved=resolved,
        is_main_process=is_main,
    )

    if not validation.all_valid and not resolved.allow_stale_data:
        raise ValueError(f"Data validation failed: {validation.errors}")

    # Phase 4: Initialize model
    model_result = initialize_training_model(
        config=config,
        resolved=resolved,
        data_path=data_path,
        device=device,
        is_main_process=is_main,
    )

    if model_result.model is None:
        raise RuntimeError(f"Model initialization failed: {model_result.errors}")

    # Phase 5: Build context
    context = build_train_context(
        config=config,
        resolved=resolved,
        model_result=model_result,
        data_paths=data_paths,
        save_path=save_path,
        device=device,
    )

    # Phase 6: Execute training (using TrainLoopExecutor)
    from app.training.train_loop_executor import TrainLoopExecutor

    # Note: This requires additional setup (optimizer, loaders, etc.)
    # For now, return a stub result showing the pattern
    logger.info(
        f"Components initialized successfully:\n"
        f"  - Model: {model_result.model_version} ({model_result.effective_blocks} blocks, "
        f"{model_result.effective_filters} filters)\n"
        f"  - Policy size: {model_result.policy_size}\n"
        f"  - Board size: {model_result.board_size}\n"
        f"  - Device: {device}\n"
        f"  - Data paths: {len(data_paths)} file(s)"
    )

    # TODO: Complete integration with optimizer, loaders, and training loop
    return {
        "status": "components_initialized",
        "model_version": model_result.model_version,
        "board_size": model_result.board_size,
        "policy_size": model_result.policy_size,
    }
