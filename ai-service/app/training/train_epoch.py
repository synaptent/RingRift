"""Epoch-level training logic for RingRift Neural Network AI.

December 2025: Extracted from train.py to improve modularity.

This module provides epoch-level training logic, including:
- Training epoch execution
- Validation loop
- LR scheduling
- Early stopping checks
- Epoch statistics and logging

Usage:
    from app.training.train_epoch import EpochContext, run_training_epoch

    context = EpochContext(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
    )
    result = run_training_epoch(context, epoch=0)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

import torch
import torch.nn as nn

from app.training.train_step import (
    TrainStepConfig,
    TrainStepContext,
    TrainStepResult,
    parse_batch,
    run_training_step,
    transfer_batch_to_device,
)

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration and Context
# =============================================================================


@dataclass
class EpochConfig:
    """Configuration for epoch-level training."""

    # Basic training
    epochs: int = 20
    batch_size: int = 512

    # Validation
    validate_every_n_epochs: int = 1

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-5

    # Logging
    log_interval: int = 100
    log_memory_usage: bool = False

    # Distributed training
    distributed: bool = False

    # Learning rate scheduling
    use_plateau_scheduler: bool = False

    # Event publishing
    publish_events: bool = True

    # Board configuration (for event payloads)
    board_type: str = "hex8"
    num_players: int = 2


@dataclass
class EpochContext:
    """Context for epoch-level training.

    Holds all the state needed to run training epochs.
    """

    # Core components
    model: nn.Module
    optimizer: "Optimizer"
    train_loader: Any  # DataLoader or StreamingDataLoader
    val_loader: Any | None
    device: torch.device
    config: EpochConfig

    # LR schedulers
    epoch_scheduler: "_LRScheduler | None" = None
    plateau_scheduler: Any | None = None

    # Mixed precision
    grad_scaler: Any | None = None
    amp_enabled: bool = False
    amp_dtype: torch.dtype = torch.float16

    # Enhancements
    training_facade: Any | None = None
    hard_example_miner: Any | None = None
    eval_feedback_handler: Any | None = None
    calibration_tracker: Any | None = None
    hot_buffer: Any | None = None

    # Fault tolerance
    training_breaker: Any | None = None
    adaptive_clipper: Any | None = None
    gradient_surgeon: Any | None = None

    # Distributed training
    dist_metrics: Any | None = None

    # Quality training
    quality_trainer: Any | None = None

    # Streaming loader info
    is_streaming: bool = False
    has_mp_values: bool = False

    # Step config (derived from epoch config)
    step_config: TrainStepConfig | None = None

    def __post_init__(self):
        """Set up step config from epoch config."""
        if self.step_config is None:
            self.step_config = TrainStepConfig(
                use_mixed_precision=self.amp_enabled,
                amp_dtype=self.amp_dtype,
            )

    def create_step_context(self) -> TrainStepContext:
        """Create a TrainStepContext from this epoch context."""
        return TrainStepContext(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            config=self.step_config,
            grad_scaler=self.grad_scaler,
            adaptive_clipper=self.adaptive_clipper,
            gradient_surgeon=self.gradient_surgeon,
            training_breaker=self.training_breaker,
            enhancements_manager=None,  # Set by caller if needed
            training_facade=self.training_facade,
            hard_example_miner=self.hard_example_miner,
            quality_trainer=self.quality_trainer,
            hot_buffer=self.hot_buffer,
        )


@dataclass
class EpochResult:
    """Result of a training epoch."""

    epoch: int
    train_loss: float
    val_loss: float | None
    policy_accuracy: float | None = None
    learning_rate: float = 0.0
    duration_seconds: float = 0.0
    batches_processed: int = 0
    batches_skipped: int = 0
    grad_norm_mean: float | None = None
    should_stop: bool = False
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'epoch': self.epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'policy_accuracy': self.policy_accuracy,
            'lr': self.learning_rate,
            'duration': self.duration_seconds,
            'batches_processed': self.batches_processed,
            'batches_skipped': self.batches_skipped,
        }


@dataclass
class EarlyStopState:
    """State for early stopping tracking."""

    best_loss: float = float('inf')
    epochs_without_improvement: int = 0
    best_epoch: int = 0

    def update(self, loss: float, epoch: int, min_delta: float = 1e-5) -> bool:
        """Update state and return True if improved."""
        if loss < self.best_loss - min_delta:
            self.best_loss = loss
            self.epochs_without_improvement = 0
            self.best_epoch = epoch
            return True
        else:
            self.epochs_without_improvement += 1
            return False

    def should_stop(self, patience: int) -> bool:
        """Check if training should stop."""
        return self.epochs_without_improvement >= patience


# =============================================================================
# Training Loop
# =============================================================================


def run_train_loop(
    context: EpochContext,
    step_context: TrainStepContext,
    epoch: int,
) -> tuple[float, int, int, list[float]]:
    """Run the training loop for one epoch.

    Args:
        context: Epoch context
        step_context: Step context for batch processing
        epoch: Current epoch number

    Returns:
        Tuple of (avg_loss, batches_processed, batches_skipped, grad_norms)
    """
    context.model.train()
    total_loss = torch.tensor(0.0, device=context.device)
    batches_processed = 0
    batches_skipped = 0
    grad_norms = []

    # Create data iterator
    if context.is_streaming:
        if context.has_mp_values:
            data_iter = context.train_loader.iter_with_mp()
        else:
            data_iter = iter(context.train_loader)
    else:
        data_iter = iter(context.train_loader)

    # Get total batches for is_last_batch check
    try:
        total_batches = len(context.train_loader)
    except TypeError:
        total_batches = None  # Streaming loader may not have len

    for batch_idx, batch_data in enumerate(data_iter):
        is_last = total_batches is not None and batch_idx == total_batches - 1

        result = run_training_step(
            context=step_context,
            raw_batch=batch_data,
            batch_idx=batch_idx,
            is_streaming=context.is_streaming,
            has_mp_values=context.has_mp_values,
            is_last_batch=is_last,
        )

        if result.skipped:
            batches_skipped += 1
            continue

        total_loss += result.loss
        batches_processed += 1

        if result.grad_norm is not None:
            grad_norms.append(result.grad_norm)

        # Periodic logging
        if batch_idx > 0 and batch_idx % context.config.log_interval == 0:
            avg_so_far = total_loss.item() / batches_processed if batches_processed > 0 else 0
            logger.info(
                f"  Epoch {epoch+1}, Batch {batch_idx}: "
                f"avg_loss={avg_so_far:.4f}, skipped={batches_skipped}"
            )

    avg_loss = total_loss.item() / batches_processed if batches_processed > 0 else 0.0
    return avg_loss, batches_processed, batches_skipped, grad_norms


# =============================================================================
# Validation Loop
# =============================================================================


def run_validation_loop(
    context: EpochContext,
) -> tuple[float, float]:
    """Run validation loop.

    Args:
        context: Epoch context

    Returns:
        Tuple of (avg_val_loss, policy_accuracy)
    """
    if context.val_loader is None:
        return 0.0, 0.0

    context.model.eval()
    val_loss = torch.tensor(0.0, device=context.device)
    val_batches = 0
    policy_correct = 0
    policy_total = 0

    value_criterion = nn.MSELoss()

    with torch.no_grad():
        for batch_data in context.val_loader:
            # Parse batch
            batch = parse_batch(batch_data, is_streaming=False)
            batch = transfer_batch_to_device(batch, context.device)

            # Forward pass
            with torch.amp.autocast('cuda', enabled=context.amp_enabled, dtype=context.amp_dtype):
                out = context.model(batch.features, batch.globals_vec)

                if isinstance(out, tuple) and len(out) >= 2:
                    value_pred, policy_pred = out[:2]
                else:
                    value_pred, policy_pred = out

                # Value loss
                if value_pred.ndim == 2:
                    v_loss = value_criterion(value_pred[:, 0], batch.value_targets.reshape(-1))
                else:
                    v_loss = value_criterion(value_pred.reshape(-1), batch.value_targets.reshape(-1))

                # Policy accuracy (top-1)
                policy_valid = batch.policy_targets.sum(dim=1) > 0
                if torch.any(policy_valid):
                    pred_moves = policy_pred[policy_valid].argmax(dim=1)
                    target_moves = batch.policy_targets[policy_valid].argmax(dim=1)
                    correct = (pred_moves == target_moves).sum().item()
                    policy_correct += correct
                    policy_total += policy_valid.sum().item()

            val_loss += v_loss
            val_batches += 1

    avg_val_loss = val_loss.item() / val_batches if val_batches > 0 else 0.0
    policy_accuracy = policy_correct / policy_total if policy_total > 0 else 0.0

    return avg_val_loss, policy_accuracy


# =============================================================================
# LR Scheduling
# =============================================================================


def step_schedulers(
    context: EpochContext,
    val_loss: float,
    epoch: int,
) -> None:
    """Step learning rate schedulers.

    Args:
        context: Epoch context
        val_loss: Validation loss for plateau scheduler
        epoch: Current epoch number
    """
    if context.epoch_scheduler is not None:
        context.epoch_scheduler.step()
    elif context.plateau_scheduler is not None:
        context.plateau_scheduler.step(val_loss)

    # Apply curriculum LR scaling from training facade
    if context.training_facade is not None:
        try:
            if hasattr(context.training_facade.config, 'enable_curriculum_lr'):
                if context.training_facade.config.enable_curriculum_lr:
                    scale = context.training_facade.get_curriculum_lr_scale()
                    if abs(scale - 1.0) > 0.01:
                        base_lr = context.optimizer.param_groups[0]['lr']
                        for param_group in context.optimizer.param_groups:
                            param_group['lr'] = base_lr * scale
        except (AttributeError, ValueError) as e:
            logger.debug(f"Curriculum LR scaling error: {e}")

    # Apply evaluation feedback LR adjustment
    if context.eval_feedback_handler is not None:
        try:
            if context.eval_feedback_handler.should_adjust_lr():
                context.eval_feedback_handler.apply_lr_adjustment(current_epoch=epoch)
        except (AttributeError, ValueError) as e:
            logger.debug(f"Evaluation feedback LR adjustment error: {e}")


# =============================================================================
# Epoch Statistics
# =============================================================================


def log_epoch_stats(
    context: EpochContext,
    result: EpochResult,
) -> None:
    """Log epoch statistics.

    Args:
        context: Epoch context
        result: Epoch result
    """
    config = context.config

    # Skip logging on non-main processes in distributed mode
    if config.distributed:
        try:
            from app.training.distributed import is_main_process
            if not is_main_process():
                return
        except ImportError:
            pass

    log_msg = (
        f"Epoch [{result.epoch+1}/{config.epochs}], "
        f"Train Loss: {result.train_loss:.4f}"
    )

    if result.val_loss is not None:
        log_msg += f", Val Loss: {result.val_loss:.4f}"

    if result.policy_accuracy is not None:
        log_msg += f", Policy Acc: {result.policy_accuracy:.1%}"

    logger.info(log_msg)
    logger.info(f"  Current LR: {result.learning_rate:.6f}")

    # Log hot buffer stats
    if context.hot_buffer is not None:
        try:
            stats = context.hot_buffer.get_statistics()
            logger.info(
                f"  Hot Buffer: {stats['game_count']}/{stats['max_size']} games"
            )
        except (AttributeError, KeyError):
            pass

    # Log training facade stats
    if context.training_facade is not None:
        try:
            facade_stats = context.training_facade.on_epoch_end()
            if facade_stats.get('mining_active', False):
                logger.info(
                    f"  [Training Facade] "
                    f"tracked={facade_stats.get('tracked_samples', 0)}, "
                    f"hard_frac={facade_stats.get('hard_examples_fraction', 0):.1%}"
                )
        except (AttributeError, ValueError):
            pass


def publish_epoch_event(
    context: EpochContext,
    result: EpochResult,
) -> None:
    """Publish epoch completion event.

    Args:
        context: Epoch context
        result: Epoch result
    """
    if not context.config.publish_events:
        return

    try:
        from app.coordination.event_router import get_router, DataEvent, DataEventType

        router = get_router()
        event_payload = {
            "epoch": result.epoch + 1,
            "total_epochs": context.config.epochs,
            "train_loss": result.train_loss,
            "val_loss": result.val_loss,
            "policy_accuracy": result.policy_accuracy,
            "lr": result.learning_rate,
            "config": f"{context.config.board_type}_{context.config.num_players}p",
        }

        router.publish_sync(DataEvent(
            event_type=DataEventType.TRAINING_PROGRESS,
            payload=event_payload,
            source="train_epoch",
        ))
    except (ImportError, RuntimeError, ConnectionError) as e:
        logger.debug(f"Failed to publish epoch event: {e}")


def check_overfitting(
    train_loss: float,
    val_loss: float,
    epoch: int,
    threshold: float = 0.25,
    min_epochs: int = 3,
) -> bool:
    """Check for overfitting based on train/val divergence.

    Args:
        train_loss: Training loss
        val_loss: Validation loss
        epoch: Current epoch
        threshold: Divergence threshold (0.25 = 25%)
        min_epochs: Minimum epochs before checking

    Returns:
        True if overfitting detected
    """
    if train_loss <= 0 or epoch < min_epochs:
        return False

    divergence = (val_loss - train_loss) / train_loss
    if divergence > threshold:
        logger.warning(
            f"Overfitting detected: {divergence*100:.1f}% divergence "
            f"(train={train_loss:.4f}, val={val_loss:.4f})"
        )
        return True
    return False


# =============================================================================
# Main Epoch Function
# =============================================================================


def run_training_epoch(
    context: EpochContext,
    epoch: int,
    early_stop_state: EarlyStopState | None = None,
) -> EpochResult:
    """Run a single training epoch.

    This is the main entry point for epoch-level training.

    Args:
        context: Epoch context with model, optimizer, data, etc.
        epoch: Current epoch number (0-indexed)
        early_stop_state: Optional early stopping state

    Returns:
        EpochResult with epoch statistics
    """
    start_time = time.time()
    config = context.config

    # Create step context
    step_context = context.create_step_context()

    # Run training loop
    train_loss, batches_processed, batches_skipped, grad_norms = run_train_loop(
        context, step_context, epoch
    )

    # Run validation
    val_loss = None
    policy_accuracy = None
    if context.val_loader is not None and (epoch + 1) % config.validate_every_n_epochs == 0:
        val_loss, policy_accuracy = run_validation_loop(context)

    # Step schedulers
    if val_loss is not None:
        step_schedulers(context, val_loss, epoch)

    # Get current learning rate
    current_lr = context.optimizer.param_groups[0]['lr']

    # Compute gradient norm mean
    grad_norm_mean = sum(grad_norms) / len(grad_norms) if grad_norms else None

    # Check early stopping
    should_stop = False
    stop_reason = None
    if early_stop_state is not None and val_loss is not None:
        early_stop_state.update(val_loss, epoch, config.min_delta)
        if early_stop_state.should_stop(config.patience):
            should_stop = True
            stop_reason = f"No improvement for {config.patience} epochs"

    # Check overfitting
    if val_loss is not None:
        check_overfitting(train_loss, val_loss, epoch)

    duration = time.time() - start_time

    result = EpochResult(
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        policy_accuracy=policy_accuracy,
        learning_rate=current_lr,
        duration_seconds=duration,
        batches_processed=batches_processed,
        batches_skipped=batches_skipped,
        grad_norm_mean=grad_norm_mean,
        should_stop=should_stop,
        stop_reason=stop_reason,
    )

    # Log and publish
    log_epoch_stats(context, result)
    publish_epoch_event(context, result)

    return result


def run_all_epochs(
    context: EpochContext,
    start_epoch: int = 0,
    early_stop_state: EarlyStopState | None = None,
) -> list[EpochResult]:
    """Run all training epochs.

    Args:
        context: Epoch context
        start_epoch: Starting epoch (for resuming)
        early_stop_state: Optional early stopping state

    Returns:
        List of EpochResult for each epoch
    """
    if early_stop_state is None:
        early_stop_state = EarlyStopState()

    results = []
    for epoch in range(start_epoch, context.config.epochs):
        result = run_training_epoch(context, epoch, early_stop_state)
        results.append(result)

        if result.should_stop:
            logger.info(f"Early stopping: {result.stop_reason}")
            break

    return results
