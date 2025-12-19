"""Training setup utilities for RingRift AI.

This module provides factory functions to reduce boilerplate in train.py
by consolidating component initialization logic.

December 2025: Extracted from train.py to improve modularity.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from app.training.training_enhancements import (
        TrainingAnomalyDetector,
        AdaptiveGradientClipper,
    )
    from app.training.checkpoint_unified import GracefulShutdownHandler

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
    training_breaker: Optional[Any] = None
    anomaly_detector: Optional[Any] = None
    adaptive_clipper: Optional[Any] = None
    shutdown_handler: Optional[Any] = None
    gradient_clip_mode: str = 'fixed'
    fixed_clip_norm: float = 1.0


def setup_fault_tolerance(
    config: FaultToleranceConfig,
    distributed: bool = False,
    is_main_process_fn: Optional[Callable[[], bool]] = None,
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
    is_main_process_fn: Optional[Callable[[], bool]] = None,
) -> Optional[Any]:
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
    last_good_checkpoint_path: Optional[str] = None
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


__all__ = [
    # Configuration
    'FaultToleranceConfig',
    'FaultToleranceComponents',
    # Setup functions
    'setup_fault_tolerance',
    'setup_graceful_shutdown',
    # Helpers
    'get_device',
    'compute_effective_lr',
    # State management
    'TrainingState',
]
