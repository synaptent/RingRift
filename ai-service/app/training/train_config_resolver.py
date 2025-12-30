"""Training configuration resolver for RingRift Neural Network AI.

December 2025: Extracted from train.py to improve modularity.

This module provides the TrainConfigResolver class which resolves all training
parameters from config, CLI arguments, and defaults with proper precedence.

Usage:
    from app.training.train_config_resolver import TrainConfigResolver

    resolver = TrainConfigResolver(config)
    resolved = resolver.resolve(
        early_stopping_patience=None,  # Will use config or default
        warmup_epochs=3,  # Override
    )
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import torch

from app.config.thresholds import (
    EARLY_STOPPING_PATIENCE,
    ELO_PATIENCE,
    MIN_TRAINING_EPOCHS,
)
from app.training.train_context import ResolvedConfig

if TYPE_CHECKING:
    from app.training.train_config import TrainConfig

logger = logging.getLogger(__name__)


class TrainConfigResolver:
    """Resolves training parameters with proper precedence.

    Precedence order (highest to lowest):
    1. Explicit overrides passed to resolve()
    2. Values from TrainConfig
    3. Default values from thresholds.py

    Example:
        resolver = TrainConfigResolver(config)
        resolved = resolver.resolve(
            early_stopping_patience=None,  # Use config or default
            warmup_epochs=5,  # Override value
        )
    """

    def __init__(self, config: "TrainConfig"):
        """Initialize the resolver with a training config.

        Args:
            config: The base training configuration
        """
        self.config = config

    def resolve(self, **overrides: Any) -> ResolvedConfig:
        """Resolve all parameters with precedence: override > config > default.

        Args:
            **overrides: Parameter overrides. None values will fall through
                to config or defaults.

        Returns:
            A ResolvedConfig with all parameters resolved.
        """
        # Start with defaults
        resolved = ResolvedConfig()

        # Build resolved values for each parameter
        resolved_dict = asdict(resolved)

        # Apply config values where available
        for key in resolved_dict:
            config_value = getattr(self.config, key, None)
            if config_value is not None:
                resolved_dict[key] = config_value

        # Apply explicit overrides (non-None values only)
        for key, value in overrides.items():
            if value is not None and key in resolved_dict:
                resolved_dict[key] = value

        # Handle special cases with thresholds.py defaults
        if resolved_dict.get("early_stopping_patience") == ResolvedConfig.early_stopping_patience:
            # Use threshold constant if not overridden
            config_val = getattr(self.config, "early_stopping_patience", None)
            if config_val is None and overrides.get("early_stopping_patience") is None:
                resolved_dict["early_stopping_patience"] = EARLY_STOPPING_PATIENCE

        if resolved_dict.get("elo_early_stopping_patience") == ResolvedConfig.elo_early_stopping_patience:
            config_val = getattr(self.config, "elo_early_stopping_patience", None)
            if config_val is None and overrides.get("elo_early_stopping_patience") is None:
                resolved_dict["elo_early_stopping_patience"] = ELO_PATIENCE

        # Validate and return
        return ResolvedConfig(**resolved_dict)

    def resolve_device(
        self,
        distributed: bool = False,
        local_rank: int = -1,
    ) -> torch.device:
        """Resolve device selection.

        Args:
            distributed: Whether distributed training is enabled
            local_rank: Local rank for distributed training

        Returns:
            The selected torch device
        """
        if distributed:
            # In distributed mode, use the local_rank device
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{local_rank}")
                torch.cuda.set_device(device)
            else:
                device = torch.device("cpu")
            logger.debug(f"Distributed device: {device} (rank {local_rank})")
        else:
            # Standard single-device selection
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        return device

    def get_amp_dtype(self, amp_dtype_str: str = "bfloat16") -> torch.dtype:
        """Get the AMP dtype from string.

        Args:
            amp_dtype_str: String representation ("bfloat16" or "float16")

        Returns:
            The corresponding torch dtype
        """
        if amp_dtype_str == "bfloat16":
            return torch.bfloat16
        return torch.float16

    def get_world_size(self, distributed: bool = False) -> int:
        """Get the world size for distributed training.

        Args:
            distributed: Whether distributed training is enabled

        Returns:
            World size (1 for non-distributed)
        """
        if not distributed:
            return 1

        try:
            if torch.distributed.is_initialized():
                return torch.distributed.get_world_size()
        except RuntimeError:
            # torch.distributed not initialized or failed - fall back to single process
            pass

        return 1

    def scale_learning_rate(
        self,
        base_lr: float,
        world_size: int,
        scale_mode: str = "linear",
    ) -> float:
        """Scale learning rate for distributed training.

        Args:
            base_lr: Base learning rate
            world_size: Number of processes
            scale_mode: Scaling mode ("linear" or "sqrt")

        Returns:
            Scaled learning rate
        """
        if world_size <= 1:
            return base_lr

        if scale_mode == "sqrt":
            return base_lr * (world_size ** 0.5)
        else:  # linear
            return base_lr * world_size

    def get_effective_batch_size(
        self,
        batch_size: int,
        world_size: int = 1,
    ) -> int:
        """Get effective batch size across all processes.

        Args:
            batch_size: Per-process batch size
            world_size: Number of processes

        Returns:
            Total effective batch size
        """
        return batch_size * world_size

    def log_improvements_status(
        self,
        resolved: ResolvedConfig,
        is_main_process: bool = True,
    ) -> list[str]:
        """Log 2024-12 Training Improvements status.

        Args:
            resolved: The resolved configuration
            is_main_process: Whether this is the main process

        Returns:
            List of enabled improvement names
        """
        improvements_enabled = []

        if resolved.spectral_norm:
            improvements_enabled.append("spectral_norm")
        if resolved.cyclic_lr:
            improvements_enabled.append(f"cyclic_lr(period={resolved.cyclic_lr_period})")
        if resolved.mixed_precision:
            improvements_enabled.append(f"mixed_precision({resolved.amp_dtype})")
        if resolved.value_whitening:
            improvements_enabled.append("value_whitening")
        if resolved.ema:
            improvements_enabled.append(f"ema(decay={resolved.ema_decay})")
        if resolved.stochastic_depth:
            improvements_enabled.append(f"stochastic_depth(p={resolved.stochastic_depth_prob})")
        if resolved.adaptive_warmup:
            improvements_enabled.append("adaptive_warmup")
        if resolved.hard_example_mining:
            improvements_enabled.append(f"hard_example_mining(top_k={resolved.hard_example_top_k})")
        if resolved.enable_outcome_weighted_policy:
            improvements_enabled.append(f"outcome_weighted_policy(scale={resolved.outcome_weight_scale})")
        if resolved.enable_quality_weighting:
            improvements_enabled.append(f"quality_weighting(blend={resolved.quality_weight_blend})")
        if resolved.enable_checkpoint_averaging:
            improvements_enabled.append(f"checkpoint_averaging(n={resolved.num_checkpoints_to_average})")

        if improvements_enabled and is_main_process:
            logger.info(f"2024-12 Training Improvements enabled: {', '.join(improvements_enabled)}")

        return improvements_enabled

    def log_early_stopping_config(
        self,
        resolved: ResolvedConfig,
        is_main_process: bool = True,
    ) -> None:
        """Log early stopping configuration.

        Args:
            resolved: The resolved configuration
            is_main_process: Whether this is the main process
        """
        if not is_main_process:
            return

        if resolved.early_stopping_patience > 0:
            elo_info = ""
            if resolved.elo_early_stopping_patience > 0:
                elo_info = (
                    f", Elo patience: {resolved.elo_early_stopping_patience} "
                    f"(min improvement: {resolved.elo_min_improvement})"
                )
            logger.info(
                f"Early stopping enabled with loss patience: "
                f"{resolved.early_stopping_patience}{elo_info}"
            )

        if resolved.warmup_epochs > 0:
            logger.info(f"LR warmup enabled for {resolved.warmup_epochs} epochs")

        if resolved.lr_scheduler in ("cosine", "cosine-warm-restarts"):
            logger.info(f"LR scheduler: {resolved.lr_scheduler} (min_lr={resolved.lr_min})")
            if resolved.lr_scheduler == "cosine-warm-restarts":
                logger.info(f"  T_0={resolved.lr_t0}, T_mult={resolved.lr_t_mult}")

    def validate_min_epochs(
        self,
        resolved: ResolvedConfig,
        is_main_process: bool = True,
    ) -> bool:
        """Validate minimum epochs configuration.

        Args:
            resolved: The resolved configuration
            is_main_process: Whether this is the main process

        Returns:
            True if configuration is valid
        """
        if self.config.epochs < MIN_TRAINING_EPOCHS:
            if is_main_process:
                logger.warning(
                    f"Requested epochs ({self.config.epochs}) is below minimum "
                    f"({MIN_TRAINING_EPOCHS}). Training may stop early."
                )
            return False
        return True


# =============================================================================
# Factory functions
# =============================================================================


def resolve_training_config(
    config: "TrainConfig",
    **kwargs: Any,
) -> ResolvedConfig:
    """Convenience function to resolve training configuration.

    Args:
        config: The training configuration
        **kwargs: Parameter overrides

    Returns:
        Resolved configuration
    """
    resolver = TrainConfigResolver(config)
    return resolver.resolve(**kwargs)


def resolve_device(
    distributed: bool = False,
    local_rank: int = -1,
) -> torch.device:
    """Convenience function to resolve device.

    Args:
        distributed: Whether distributed training is enabled
        local_rank: Local rank for distributed training

    Returns:
        The selected torch device
    """
    if distributed:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    return device
