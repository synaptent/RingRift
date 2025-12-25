"""
Training checkpoint utilities.

Extracted from train.py (December 2025) to reduce module size.
Provides high-level checkpoint management for training loops.

Note: For low-level checkpoint operations, use checkpoint_unified.py directly.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """State loaded from or saved to a checkpoint."""

    epoch: int
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any] | None = None
    scheduler_state_dict: dict[str, Any] | None = None
    best_val_loss: float | None = None
    best_policy_acc: float | None = None
    training_config: dict[str, Any] | None = None
    ema_state_dict: dict[str, Any] | None = None


class AsyncCheckpointer:
    """Non-blocking checkpoint saver using background threads.

    Provides 5-10% speedup by writing checkpoints asynchronously.
    """

    def __init__(self, max_pending: int = 2):
        """Initialize async checkpointer.

        Args:
            max_pending: Maximum number of pending saves before blocking
        """
        self.max_pending = max_pending
        self._pending: list[threading.Thread] = []
        self._lock = threading.Lock()

    def save_async(
        self,
        state_dict: dict[str, Any],
        path: str,
        callback: callable | None = None,
    ) -> None:
        """Save checkpoint asynchronously.

        Args:
            state_dict: Checkpoint state to save
            path: Path to save checkpoint
            callback: Optional callback after save completes
        """
        # Clean up completed threads
        self._cleanup_completed()

        # If too many pending, wait for one to complete
        while len(self._pending) >= self.max_pending:
            self._pending[0].join()
            self._cleanup_completed()

        def _save():
            try:
                # Create parent directory if needed
                os.makedirs(os.path.dirname(path), exist_ok=True)

                # Save to temporary file first, then rename (atomic)
                tmp_path = path + ".tmp"
                torch.save(state_dict, tmp_path)
                os.replace(tmp_path, path)

                if callback:
                    callback()
            except Exception as e:
                logger.error(f"Async checkpoint save failed: {e}")

        thread = threading.Thread(target=_save, daemon=True)
        thread.start()

        with self._lock:
            self._pending.append(thread)

    def _cleanup_completed(self) -> None:
        """Remove completed threads from pending list."""
        with self._lock:
            self._pending = [t for t in self._pending if t.is_alive()]

    def wait_all(self) -> None:
        """Wait for all pending saves to complete."""
        for thread in self._pending:
            thread.join()
        self._pending.clear()


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    best_val_loss: float | None = None,
    best_policy_acc: float | None = None,
    training_config: dict[str, Any] | None = None,
    ema_model: nn.Module | None = None,
    async_checkpointer: AsyncCheckpointer | None = None,
) -> None:
    """Save a training checkpoint.

    Args:
        path: Path to save checkpoint
        model: Model to checkpoint
        optimizer: Optimizer state (optional)
        scheduler: LR scheduler state (optional)
        epoch: Current epoch number
        best_val_loss: Best validation loss seen
        best_policy_acc: Best policy accuracy seen
        training_config: Training configuration dict
        ema_model: EMA model if using exponential moving average
        async_checkpointer: If provided, save asynchronously
    """
    # Handle DDP wrapped models
    model_to_save = model.module if hasattr(model, "module") else model

    state_dict = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
    }

    if optimizer is not None:
        state_dict["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        state_dict["scheduler_state_dict"] = scheduler.state_dict()

    if best_val_loss is not None:
        state_dict["best_val_loss"] = best_val_loss

    if best_policy_acc is not None:
        state_dict["best_policy_acc"] = best_policy_acc

    if training_config is not None:
        state_dict["training_config"] = training_config

    if ema_model is not None:
        ema_to_save = ema_model.module if hasattr(ema_model, "module") else ema_model
        state_dict["ema_state_dict"] = ema_to_save.state_dict()

    if async_checkpointer is not None:
        async_checkpointer.save_async(state_dict, path)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> CheckpointState:
    """Load a training checkpoint.

    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to restore state (optional)
        scheduler: LR scheduler to restore state (optional)
        device: Device to load checkpoint to
        strict: If True, require exact key match for model

    Returns:
        CheckpointState with loaded values
    """
    from app.utils.torch_utils import safe_load_checkpoint
    checkpoint = safe_load_checkpoint(path, map_location=device)

    # Load model state
    model_state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))

    # Remove 'module.' prefix if present
    cleaned_state = {}
    for k, v in model_state.items():
        if k.startswith("module."):
            cleaned_state[k[7:]] = v
        else:
            cleaned_state[k] = v

    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(cleaned_state, strict=strict)

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return CheckpointState(
        epoch=checkpoint.get("epoch", 0),
        model_state_dict=cleaned_state,
        optimizer_state_dict=checkpoint.get("optimizer_state_dict"),
        scheduler_state_dict=checkpoint.get("scheduler_state_dict"),
        best_val_loss=checkpoint.get("best_val_loss"),
        best_policy_acc=checkpoint.get("best_policy_acc"),
        training_config=checkpoint.get("training_config"),
        ema_state_dict=checkpoint.get("ema_state_dict"),
    )


def get_latest_checkpoint(checkpoint_dir: str, prefix: str = "epoch_") -> str | None:
    """Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        prefix: Checkpoint filename prefix

    Returns:
        Path to latest checkpoint, or None if not found
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith(prefix) and f.endswith(".pt"):
            try:
                # Extract epoch number
                epoch_str = f[len(prefix) :].replace(".pt", "")
                epoch = int(epoch_str)
                checkpoints.append((epoch, os.path.join(checkpoint_dir, f)))
            except ValueError:
                continue

    if not checkpoints:
        return None

    # Return checkpoint with highest epoch
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_last: int = 5,
    keep_best: bool = True,
    prefix: str = "epoch_",
) -> int:
    """Remove old checkpoints, keeping only the most recent.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of recent checkpoints to keep
        keep_best: If True, also keep 'best_model.pt'
        prefix: Checkpoint filename prefix

    Returns:
        Number of checkpoints removed
    """
    if not os.path.isdir(checkpoint_dir):
        return 0

    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith(prefix) and f.endswith(".pt"):
            try:
                epoch_str = f[len(prefix) :].replace(".pt", "")
                epoch = int(epoch_str)
                checkpoints.append((epoch, os.path.join(checkpoint_dir, f)))
            except ValueError:
                continue

    if len(checkpoints) <= keep_last:
        return 0

    # Sort by epoch, keep last N
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    to_remove = checkpoints[keep_last:]

    removed = 0
    for epoch, path in to_remove:
        try:
            os.remove(path)
            removed += 1
            logger.debug(f"Removed old checkpoint: {path}")
        except OSError as e:
            logger.warning(f"Failed to remove checkpoint {path}: {e}")

    return removed


class EarlyStopper:
    """Early stopping helper for training loops."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """Initialize early stopper.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss-like metrics, 'max' for accuracy-like
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: float | None = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """Check if training should stop.

        Args:
            value: Current metric value

        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self) -> None:
        """Reset the early stopper state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False
