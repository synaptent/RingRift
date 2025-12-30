"""Training loop executor for RingRift training.

December 2025: Extracted from train.py to improve modularity.

This module provides the TrainLoopExecutor class which orchestrates the main
training loop using all the extracted components:
- TrainContext for state management
- TrainConfigResolver for parameter resolution
- DataValidator for data validation
- ModelInitializer for model setup
- PostEpochHandler for post-epoch processing
- TrainCleanupHandler for cleanup

Usage:
    from app.training.train_loop_executor import TrainLoopExecutor

    executor = TrainLoopExecutor(context)
    result = executor.run()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from app.training.post_epoch_handler import (
    PostEpochHandler,
    PostEpochConfig,
    EpochMetrics,
)
from app.training.train_cleanup_handler import (
    TrainCleanupHandler,
    CleanupConfig,
    TrainingResult,
)

if TYPE_CHECKING:
    from app.training.train_context import TrainContext

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ExecutorConfig:
    """Configuration for the training loop executor.

    Controls how the training loop runs.
    """

    # Validation
    validate_every_n_epochs: int = 1

    # Logging
    log_interval: int = 100  # Log every N batches
    log_epoch_summary: bool = True

    # Checkpointing
    checkpoint_interval: int = 5

    # Event emission
    emit_events: bool = True

    # Distributed
    sync_batch_norm: bool = True


# =============================================================================
# Training Loop Executor
# =============================================================================


class TrainLoopExecutor:
    """Orchestrates the main training loop.

    Uses the extracted components to run a complete training session:
    1. Per-epoch training with train_epoch()
    2. Validation
    3. Post-epoch handling (checkpointing, early stopping, etc.)
    4. Cleanup on completion or error

    Example:
        context = TrainContext.from_config(config, data_path, save_path)
        # ... initialize model, optimizer, etc. ...

        executor = TrainLoopExecutor(context)
        result = executor.run()

        print(f"Training completed: {result.epochs_completed} epochs")
        print(f"Best validation loss: {result.best_val_loss:.4f}")
    """

    def __init__(
        self,
        context: "TrainContext",
        config: ExecutorConfig | None = None,
    ):
        """Initialize the executor.

        Args:
            context: Training context with all components
            config: Executor configuration
        """
        self.context = context
        self.config = config or ExecutorConfig()

        # Initialize handlers
        self._post_epoch_handler = PostEpochHandler(
            PostEpochConfig(
                checkpoint_interval=self.config.checkpoint_interval,
                checkpoint_dir=context.resolved.checkpoint_dir,
                emit_events=self.config.emit_events,
            )
        )
        self._cleanup_handler = TrainCleanupHandler(
            CleanupConfig(
                emit_events=self.config.emit_events,
            )
        )

        # Track state
        self._start_time = time.time()
        self._epoch_losses: list[dict[str, float]] = []

    def run(self) -> TrainingResult:
        """Execute the training loop.

        Returns:
            TrainingResult with all metrics
        """
        context = self.context
        completed_normally = False
        exception = None

        try:
            # Main training loop
            for epoch in range(context.config.epochs):
                context.progress.epoch = epoch

                # Set epoch for distributed sampler
                if context.train_sampler is not None:
                    context.train_sampler.set_epoch(epoch)

                # Run one epoch
                epoch_result = self._run_epoch(epoch)

                # Record epoch losses
                self._epoch_losses.append(epoch_result)
                context.progress.epoch_losses = self._epoch_losses

                # Post-epoch handling
                metrics = EpochMetrics(
                    epoch=epoch,
                    avg_train_loss=epoch_result["train_loss"],
                    avg_val_loss=epoch_result["val_loss"],
                    avg_policy_accuracy=epoch_result.get("policy_accuracy", 0.0),
                    learning_rate=epoch_result.get("lr", 0.0),
                    avg_policy_loss=epoch_result.get("policy_loss", 0.0),
                    avg_value_loss=epoch_result.get("value_loss", 0.0),
                    train_batches=epoch_result.get("train_batches", 0),
                    samples_per_second=epoch_result.get("samples_per_second", 0.0),
                    epoch_duration=epoch_result.get("epoch_duration", 0.0),
                    epoch_losses=self._epoch_losses,
                )

                post_result = self._post_epoch_handler.handle_epoch_end(
                    context=context,
                    metrics=metrics,
                )

                # Check for early stopping
                if post_result.should_stop:
                    context.progress.epochs_completed = epoch + 1
                    completed_normally = True
                    logger.info(f"Training stopped early: {post_result.stop_reason}")
                    break

                context.progress.epochs_completed = epoch + 1

            else:
                # Normal completion (no early stopping)
                completed_normally = True

                # Handle final checkpoint and averaging
                if context.is_main_process:
                    final_metrics = EpochMetrics(
                        epoch=context.config.epochs - 1,
                        avg_train_loss=self._epoch_losses[-1]["train_loss"],
                        avg_val_loss=self._epoch_losses[-1]["val_loss"],
                        avg_policy_accuracy=self._epoch_losses[-1].get("policy_accuracy", 0.0),
                        learning_rate=self._epoch_losses[-1].get("lr", 0.0),
                        epoch_losses=self._epoch_losses,
                    )
                    self._post_epoch_handler.handle_training_complete(
                        context=context,
                        metrics=final_metrics,
                    )

        except (RuntimeError, ValueError, OSError, KeyError) as e:
            exception = e
            raise

        finally:
            # Always run cleanup
            result = self._cleanup_handler.cleanup(
                context=context,
                completed_normally=completed_normally,
                exception=exception,
                result=self._build_result(completed_normally),
                start_time=self._start_time,
            )

        return result

    def _run_epoch(self, epoch: int) -> dict[str, Any]:
        """Run a single training epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with epoch metrics
        """
        context = self.context
        epoch_start = time.time()

        # Use the existing epoch training infrastructure
        epoch_context = context.create_epoch_context()

        # Train
        train_metrics = self._train_one_epoch(epoch_context, epoch)

        # Validate
        val_metrics = self._validate_one_epoch(epoch_context, epoch)

        # Update learning rate
        self._step_schedulers(context, val_metrics["val_loss"])

        # Build epoch result
        epoch_duration = time.time() - epoch_start
        train_size = (
            len(context.train_loader.dataset)
            if hasattr(context.train_loader, "dataset")
            else 0
        )

        result = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["train_loss"],
            "val_loss": val_metrics["val_loss"],
            "policy_accuracy": val_metrics.get("policy_accuracy", 0.0),
            "policy_loss": train_metrics.get("policy_loss", 0.0),
            "value_loss": train_metrics.get("value_loss", 0.0),
            "lr": context.optimizer.param_groups[0]["lr"],
            "train_batches": train_metrics.get("batches", 0),
            "epoch_duration": epoch_duration,
            "samples_per_second": train_size / epoch_duration if epoch_duration > 0 else 0,
        }

        # Log epoch summary
        if self.config.log_epoch_summary and context.is_main_process:
            logger.info(
                f"Epoch {epoch + 1}/{context.config.epochs} - "
                f"Train Loss: {result['train_loss']:.4f}, "
                f"Val Loss: {result['val_loss']:.4f}, "
                f"Policy Acc: {result['policy_accuracy']:.2%}, "
                f"LR: {result['lr']:.2e}, "
                f"Time: {epoch_duration:.1f}s"
            )

        return result

    def _train_one_epoch(
        self,
        epoch_context: Any,
        epoch: int,
    ) -> dict[str, Any]:
        """Train for one epoch.

        Args:
            epoch_context: Epoch context
            epoch: Current epoch number

        Returns:
            Training metrics
        """
        from app.training.train_epoch import train_epoch

        context = self.context
        model = context.model
        optimizer = context.optimizer
        train_loader = context.train_loader
        device = context.device

        # Training mode
        model.train()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        # Use train_epoch if available, otherwise simple loop
        try:
            result = train_epoch(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                device=device,
                epoch=epoch,
                grad_scaler=context.grad_scaler,
                amp_dtype=context.amp_dtype if context.amp_enabled else None,
                distributed=context.distributed,
            )
            return {
                "train_loss": result.get("loss", 0.0),
                "policy_loss": result.get("policy_loss", 0.0),
                "value_loss": result.get("value_loss", 0.0),
                "batches": result.get("batches", 0),
            }
        except ImportError:
            # Fallback to simple training loop
            pass

        # Simple training loop fallback
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # Move to device
            if isinstance(batch, dict):
                features = batch["features"].to(device)
                policy_target = batch["policy"].to(device)
                value_target = batch["value"].to(device)
            else:
                features, policy_target, value_target = batch
                features = features.to(device)
                policy_target = policy_target.to(device)
                value_target = value_target.to(device)

            # Forward pass
            if context.amp_enabled:
                with torch.cuda.amp.autocast(dtype=context.amp_dtype):
                    policy_out, value_out = model(features)
                    loss = self._compute_loss(
                        policy_out, value_out,
                        policy_target, value_target,
                    )

                context.grad_scaler.scale(loss).backward()
                context.grad_scaler.step(optimizer)
                context.grad_scaler.update()
            else:
                policy_out, value_out = model(features)
                loss = self._compute_loss(
                    policy_out, value_out,
                    policy_target, value_target,
                )

                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        return {
            "train_loss": avg_loss,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "batches": num_batches,
        }

    def _validate_one_epoch(
        self,
        epoch_context: Any,
        epoch: int,
    ) -> dict[str, Any]:
        """Validate for one epoch.

        Args:
            epoch_context: Epoch context
            epoch: Current epoch number

        Returns:
            Validation metrics
        """
        context = self.context
        model = context.model
        val_loader = context.val_loader
        device = context.device

        if val_loader is None:
            return {"val_loss": 0.0, "policy_accuracy": 0.0}

        model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                if isinstance(batch, dict):
                    features = batch["features"].to(device)
                    policy_target = batch["policy"].to(device)
                    value_target = batch["value"].to(device)
                else:
                    features, policy_target, value_target = batch
                    features = features.to(device)
                    policy_target = policy_target.to(device)
                    value_target = value_target.to(device)

                # Forward pass
                if context.amp_enabled:
                    with torch.cuda.amp.autocast(dtype=context.amp_dtype):
                        policy_out, value_out = model(features)
                        loss = self._compute_loss(
                            policy_out, value_out,
                            policy_target, value_target,
                        )
                else:
                    policy_out, value_out = model(features)
                    loss = self._compute_loss(
                        policy_out, value_out,
                        policy_target, value_target,
                    )

                total_loss += loss.item()

                # Policy accuracy
                pred = policy_out.argmax(dim=1)
                target = policy_target.argmax(dim=1) if policy_target.dim() > 1 else policy_target
                total_correct += (pred == target).sum().item()
                total_samples += features.size(0)
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = total_correct / max(total_samples, 1)

        return {
            "val_loss": avg_loss,
            "policy_accuracy": accuracy,
        }

    def _compute_loss(
        self,
        policy_out: torch.Tensor,
        value_out: torch.Tensor,
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined policy and value loss.

        Args:
            policy_out: Policy predictions
            value_out: Value predictions
            policy_target: Policy targets
            value_target: Value targets

        Returns:
            Combined loss tensor
        """
        import torch.nn.functional as F

        # Policy loss (cross entropy)
        if policy_target.dim() > 1:
            # Soft targets
            log_probs = F.log_softmax(policy_out, dim=1)
            policy_loss = -(policy_target * log_probs).sum(dim=1).mean()
        else:
            policy_loss = F.cross_entropy(policy_out, policy_target)

        # Value loss (MSE)
        value_loss = F.mse_loss(value_out, value_target)

        return policy_loss + value_loss

    def _step_schedulers(
        self,
        context: "TrainContext",
        val_loss: float,
    ) -> None:
        """Step learning rate schedulers.

        Args:
            context: Training context
            val_loss: Current validation loss
        """
        # Epoch scheduler
        if context.epoch_scheduler is not None:
            context.epoch_scheduler.step()

        # Plateau scheduler
        if context.plateau_scheduler is not None:
            context.plateau_scheduler.step(val_loss)

    def _build_result(self, completed_normally: bool) -> TrainingResult:
        """Build training result from current state.

        Args:
            completed_normally: Whether training completed normally

        Returns:
            TrainingResult instance
        """
        context = self.context
        progress = context.progress

        best_val_loss = min(
            (e.get("val_loss", float("inf")) for e in self._epoch_losses),
            default=float("inf"),
        )

        return TrainingResult(
            best_val_loss=best_val_loss,
            final_train_loss=self._epoch_losses[-1]["train_loss"] if self._epoch_losses else 0.0,
            final_val_loss=self._epoch_losses[-1]["val_loss"] if self._epoch_losses else 0.0,
            epochs_completed=progress.epochs_completed,
            epoch_losses=self._epoch_losses,
            completed_normally=completed_normally,
            early_stopped=False,  # Set by post_epoch_handler if applicable
            duration_seconds=time.time() - self._start_time,
            final_checkpoint_path=progress.last_good_checkpoint_path or "",
            best_model_path=context.save_path,
            final_policy_accuracy=(
                self._epoch_losses[-1].get("policy_accuracy", 0.0)
                if self._epoch_losses
                else 0.0
            ),
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_executor(
    context: "TrainContext",
    checkpoint_interval: int = 5,
    emit_events: bool = True,
    **kwargs: Any,
) -> TrainLoopExecutor:
    """Create a TrainLoopExecutor with the specified settings.

    Args:
        context: Training context
        checkpoint_interval: Epochs between checkpoints
        emit_events: Enable event emission
        **kwargs: Additional config parameters

    Returns:
        Configured TrainLoopExecutor instance
    """
    config = ExecutorConfig(
        checkpoint_interval=checkpoint_interval,
        emit_events=emit_events,
        **{k: v for k, v in kwargs.items() if hasattr(ExecutorConfig, k)},
    )
    return TrainLoopExecutor(context, config)


def run_training(
    context: "TrainContext",
    **kwargs: Any,
) -> TrainingResult:
    """Convenience function to run training.

    Args:
        context: Training context
        **kwargs: Executor configuration

    Returns:
        TrainingResult with all metrics
    """
    executor = create_executor(context, **kwargs)
    return executor.run()
