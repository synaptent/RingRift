"""Training setup utilities for RingRift AI.

This module provides factory functions to reduce boilerplate in train.py
by consolidating component initialization logic.

December 2025: Extracted from train.py to improve modularity.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from app.training.fault_tolerance import HeartbeatMonitor

logger = logging.getLogger(__name__)


# =============================================================================
# Lazy imports for optional dependencies
# =============================================================================

def _try_import_anomaly_detector():
    """Lazy import TrainingAnomalyDetector."""
    try:
        from app.training.training_enhancements import TrainingAnomalyDetector
        return TrainingAnomalyDetector
    except ImportError:
        return None


def _try_import_gradient_clipper():
    """Lazy import AdaptiveGradientClipper."""
    try:
        from app.training.training_enhancements import AdaptiveGradientClipper
        return AdaptiveGradientClipper
    except ImportError:
        return None


def _try_import_circuit_breaker():
    """Lazy import circuit breaker."""
    try:
        from app.training.exception_integration import get_training_breaker
        return get_training_breaker
    except ImportError:
        return None


def _try_import_shutdown_handler():
    """Lazy import GracefulShutdownHandler."""
    try:
        from app.training.checkpoint_unified import GracefulShutdownHandler
        return GracefulShutdownHandler
    except ImportError:
        return None


def _try_import_heartbeat_monitor():
    """Lazy import HeartbeatMonitor."""
    try:
        from app.training.fault_tolerance import HeartbeatMonitor
        return HeartbeatMonitor
    except ImportError:
        return None


# =============================================================================
# Fault Tolerance Setup
# =============================================================================

@dataclass
class FaultToleranceConfig:
    """Configuration for fault tolerance components."""
    enable_circuit_breaker: bool = True
    enable_anomaly_detection: bool = True
    enable_graceful_shutdown: bool = True
    gradient_clip_mode: str = 'adaptive'
    gradient_clip_max_norm: float = 1.0
    anomaly_spike_threshold: float = 3.0
    anomaly_gradient_threshold: float = 100.0


@dataclass
class FaultToleranceComponents:
    """Container for initialized fault tolerance components."""
    training_breaker: Any | None = None
    anomaly_detector: Any | None = None
    adaptive_clipper: Any | None = None
    shutdown_handler: Any | None = None
    gradient_clip_mode: str = 'fixed'
    fixed_clip_norm: float = 1.0


def setup_fault_tolerance(
    config: FaultToleranceConfig,
    distributed: bool = False,
    is_main_process_fn: Callable[[], bool] | None = None,
) -> FaultToleranceComponents:
    """Initialize all fault tolerance components.

    Args:
        config: Fault tolerance configuration
        distributed: Whether running in distributed mode
        is_main_process_fn: Function to check if this is the main process

    Returns:
        FaultToleranceComponents with initialized components
    """
    components = FaultToleranceComponents(
        gradient_clip_mode=config.gradient_clip_mode,
        fixed_clip_norm=config.gradient_clip_max_norm,
    )

    # Circuit breaker
    if config.enable_circuit_breaker:
        get_training_breaker = _try_import_circuit_breaker()
        if get_training_breaker:
            components.training_breaker = get_training_breaker()
            logger.info("Training circuit breaker enabled for fault tolerance")
        else:
            logger.debug("Circuit breaker not available")
    else:
        logger.info("Training circuit breaker disabled by configuration")

    # Anomaly detector
    if config.enable_anomaly_detection:
        TrainingAnomalyDetector = _try_import_anomaly_detector()
        if TrainingAnomalyDetector:
            components.anomaly_detector = TrainingAnomalyDetector(
                loss_spike_threshold=config.anomaly_spike_threshold,
                gradient_norm_threshold=config.anomaly_gradient_threshold,
                loss_window_size=100,
                halt_on_nan=False,
                max_consecutive_anomalies=10,
            )
            logger.info(
                f"Training anomaly detector enabled "
                f"(spike_threshold={config.anomaly_spike_threshold}, "
                f"gradient_threshold={config.anomaly_gradient_threshold})"
            )
        else:
            logger.debug("Anomaly detector not available")
    else:
        logger.info("Training anomaly detection disabled by configuration")

    # Gradient clipping
    if config.gradient_clip_mode == 'adaptive':
        AdaptiveGradientClipper = _try_import_gradient_clipper()
        if AdaptiveGradientClipper:
            components.adaptive_clipper = AdaptiveGradientClipper(
                initial_max_norm=config.gradient_clip_max_norm,
                percentile=90.0,
                history_size=100,
                min_clip=0.1,
                max_clip=10.0,
            )
            logger.info(
                f"Adaptive gradient clipping enabled "
                f"(initial_norm={config.gradient_clip_max_norm})"
            )
        else:
            logger.warning(
                "Adaptive gradient clipping requested but not available, using fixed"
            )
            components.gradient_clip_mode = 'fixed'

    if components.gradient_clip_mode == 'fixed':
        logger.info(f"Fixed gradient clipping enabled (max_norm={config.gradient_clip_max_norm})")

    return components


def setup_graceful_shutdown(
    checkpoint_callback: Callable[[], None],
    distributed: bool = False,
    is_main_process_fn: Callable[[], bool] | None = None,
) -> Any | None:
    """Setup graceful shutdown handler.

    Args:
        checkpoint_callback: Function to call on shutdown signal
        distributed: Whether running in distributed mode
        is_main_process_fn: Function to check if this is main process

    Returns:
        GracefulShutdownHandler or None
    """
    # Only setup on main process
    if distributed and is_main_process_fn and not is_main_process_fn():
        return None

    GracefulShutdownHandler = _try_import_shutdown_handler()
    if not GracefulShutdownHandler:
        logger.debug("GracefulShutdownHandler not available")
        return None

    handler = GracefulShutdownHandler()
    handler.setup(checkpoint_callback)
    logger.info("Graceful shutdown handler enabled")
    return handler


def setup_heartbeat_monitor(
    heartbeat_file: str | Path | None,
    heartbeat_interval: float = 30.0,
    is_main_process_fn: Callable[[], bool] | None = None,
    timeout_multiplier: float = 4.0,
) -> HeartbeatMonitor | None:
    """Initialize heartbeat monitor for training fault tolerance.

    The heartbeat monitor writes periodic updates to a file, allowing external
    systems to detect hung or crashed training processes.

    Args:
        heartbeat_file: Path to heartbeat file. If None, no monitor is created.
        heartbeat_interval: Seconds between heartbeat updates (default: 30.0).
        is_main_process_fn: Function to check if this is the main process.
            If provided and returns False, no monitor is created.
        timeout_multiplier: Multiplier for timeout threshold (default: 4.0).
            Timeout = heartbeat_interval * timeout_multiplier.

    Returns:
        HeartbeatMonitor instance if successfully created, None otherwise.

    Example:
        >>> from app.training.train_setup import setup_heartbeat_monitor
        >>> monitor = setup_heartbeat_monitor(
        ...     heartbeat_file="/tmp/training_heartbeat.json",
        ...     heartbeat_interval=30.0,
        ... )
        >>> if monitor:
        ...     # Call monitor.beat() after each epoch
        ...     monitor.beat()
        ...     # Stop when done
        ...     monitor.stop()
    """
    if heartbeat_file is None:
        return None

    # Only create on main process if function provided
    if is_main_process_fn is not None and not is_main_process_fn():
        logger.debug("Heartbeat monitor skipped (not main process)")
        return None

    HeartbeatMonitor = _try_import_heartbeat_monitor()
    if not HeartbeatMonitor:
        logger.warning("HeartbeatMonitor not available, skipping heartbeat setup")
        return None

    # Ensure parent directory exists
    heartbeat_path = Path(heartbeat_file)
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate timeout threshold
    timeout_threshold = heartbeat_interval * timeout_multiplier

    # Create and start monitor
    monitor = HeartbeatMonitor(
        heartbeat_interval=heartbeat_interval,
        timeout_threshold=timeout_threshold,
    )
    monitor.start(heartbeat_path)

    logger.info(
        f"Heartbeat monitor started: {heartbeat_file} "
        f"(interval={heartbeat_interval}s, timeout={timeout_threshold}s)"
    )

    return monitor


# =============================================================================
# Model Initialization Helpers
# =============================================================================

def get_device(local_rank: int = -1) -> torch.device:
    """Get the appropriate device for training.

    Args:
        local_rank: Local rank for distributed training (-1 for single GPU)

    Returns:
        torch.device for training
    """
    if local_rank >= 0 and torch.cuda.is_available():
        return torch.device(f'cuda:{local_rank}')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def compute_effective_lr(
    base_lr: float,
    world_size: int,
    scale_lr: bool = False,
    lr_scale_mode: str = 'linear',
) -> float:
    """Compute effective learning rate for distributed training.

    Args:
        base_lr: Base learning rate
        world_size: Number of processes in distributed training
        scale_lr: Whether to scale LR
        lr_scale_mode: 'linear' or 'sqrt' scaling

    Returns:
        Scaled learning rate
    """
    if not scale_lr or world_size <= 1:
        return base_lr

    if lr_scale_mode == 'sqrt':
        return base_lr * (world_size ** 0.5)
    else:  # linear
        return base_lr * world_size


# =============================================================================
# Training State Management
# =============================================================================

@dataclass
class TrainingState:
    """Mutable training state for checkpointing and recovery."""
    epoch: int = 0
    best_val_loss: float = float('inf')
    avg_val_loss: float = float('inf')
    last_good_checkpoint_path: str | None = None
    last_good_epoch: int = 0
    circuit_breaker_rollbacks: int = 0
    max_circuit_breaker_rollbacks: int = 3

    def can_rollback(self) -> bool:
        """Check if rollback is possible."""
        return (
            self.last_good_checkpoint_path is not None
            and self.circuit_breaker_rollbacks < self.max_circuit_breaker_rollbacks
        )

    def record_rollback(self) -> None:
        """Record a rollback attempt."""
        self.circuit_breaker_rollbacks += 1

    def update_good_checkpoint(self, path: str, epoch: int) -> None:
        """Update the last known good checkpoint."""
        self.last_good_checkpoint_path = path
        self.last_good_epoch = epoch


# =============================================================================
# Optimizer Setup
# =============================================================================

@dataclass
class OptimizerConfig:
    """Configuration for optimizer and learning rate schedulers."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_type: str = "cosine"
    warmup_epochs: int = 0
    lr_min: float = 1e-6
    lr_t0: int = 10
    lr_t_mult: int = 2
    total_epochs: int = 100
    freeze_policy: bool = False


@dataclass
class OptimizerComponents:
    """Container for initialized optimizer components."""

    optimizer: Any  # torch.optim.Optimizer
    epoch_scheduler: Any | None = None
    plateau_scheduler: Any | None = None
    trainable_params: list | None = None


def setup_parameter_freezing(
    model: Any,  # nn.Module
    freeze_policy: bool = False,
) -> list:
    """Configure which parameters to train based on freezing settings.

    When freeze_policy is True, only value head parameters are trained.
    This is useful for transfer learning or fine-tuning.

    Args:
        model: The neural network model.
        freeze_policy: If True, freeze all except value head.

    Returns:
        List of parameters to pass to the optimizer.
    """
    if not freeze_policy:
        return list(model.parameters())

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only value head parameters
    value_head_params = []
    value_patterns = ['value_fc', 'value_head', 'value_conv', 'value_bn']

    for name, param in model.named_parameters():
        if any(pattern in name.lower() for pattern in value_patterns):
            param.requires_grad = True
            value_head_params.append(param)
            logger.info(f"[freeze_policy] Unfreezing: {name}")

    if not value_head_params:
        logger.warning(
            "[freeze_policy] No value head parameters found! "
            "Check model architecture. Training all parameters."
        )
        for param in model.parameters():
            param.requires_grad = True
        return list(model.parameters())

    logger.info(f"[freeze_policy] Training only {len(value_head_params)} value head parameters")
    return value_head_params


def setup_optimizer_and_schedulers(
    model: Any,  # nn.Module
    config: OptimizerConfig,
) -> OptimizerComponents:
    """Initialize optimizer and learning rate schedulers.

    Sets up Adam optimizer with optional parameter freezing and LR scheduling.

    Args:
        model: The neural network model.
        config: Optimizer configuration.

    Returns:
        OptimizerComponents with initialized optimizer and schedulers.

    Example:
        >>> from app.training.train_setup import (
        ...     setup_optimizer_and_schedulers,
        ...     OptimizerConfig,
        ... )
        >>> config = OptimizerConfig(learning_rate=1e-3, total_epochs=50)
        >>> components = setup_optimizer_and_schedulers(model, config)
        >>> optimizer = components.optimizer
        >>> scheduler = components.epoch_scheduler
    """
    # Determine trainable parameters
    trainable_params = setup_parameter_freezing(model, config.freeze_policy)

    # Create optimizer
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Create LR scheduler
    epoch_scheduler = None
    try:
        from app.training.schedulers import create_lr_scheduler

        epoch_scheduler = create_lr_scheduler(
            optimizer,
            scheduler_type=config.scheduler_type,
            total_epochs=config.total_epochs,
            warmup_epochs=config.warmup_epochs,
            lr_min=config.lr_min,
            lr_t0=config.lr_t0,
            lr_t_mult=config.lr_t_mult,
        )
    except ImportError:
        logger.debug("Scheduler module not available, using ReduceLROnPlateau")

    # Fallback to ReduceLROnPlateau if no scheduler configured
    plateau_scheduler = None
    if epoch_scheduler is None:
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )

    return OptimizerComponents(
        optimizer=optimizer,
        epoch_scheduler=epoch_scheduler,
        plateau_scheduler=plateau_scheduler,
        trainable_params=trainable_params,
    )


__all__ = [
    # Fault tolerance
    'FaultToleranceComponents',
    'FaultToleranceConfig',
    'setup_fault_tolerance',
    'setup_graceful_shutdown',
    'setup_heartbeat_monitor',
    # Optimizer
    'OptimizerComponents',
    'OptimizerConfig',
    'setup_optimizer_and_schedulers',
    'setup_parameter_freezing',
    # State management
    'TrainingState',
    # Helpers
    'compute_effective_lr',
    'get_device',
]
