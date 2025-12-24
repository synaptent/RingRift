"""Checkpointing utilities for training.

.. deprecated:: December 2025
    Scheduled for removal in Q2 2026.

    For new code, prefer importing from :mod:`app.training.checkpoint_unified`
    which provides a unified checkpoint manager with additional features::

        from app.training.checkpoint_unified import (
            UnifiedCheckpointManager,
            get_checkpoint_manager,
        )

    This module remains available for backward compatibility and provides
    lower-level checkpoint operations.

This module provides checkpoint save/load functionality with:
- Atomic saves (temp file + rename) to prevent corruption
- Version-aware loading with backwards compatibility
- Async background saving for non-blocking I/O
- Graceful shutdown handling with emergency checkpoints

Usage:
    from app.training.checkpointing import (
        save_checkpoint,
        load_checkpoint,
        AsyncCheckpointer,
        GracefulShutdownHandler,
    )

    # Synchronous save
    save_checkpoint(model, optimizer, epoch, loss, "checkpoint.pth")

    # Async save (non-blocking)
    checkpointer = AsyncCheckpointer()
    checkpointer.save_async(model, optimizer, epoch, loss, "checkpoint.pth")
    checkpointer.shutdown()

    # Graceful shutdown
    handler = GracefulShutdownHandler()
    handler.setup(lambda: save_checkpoint(...))
"""

from __future__ import annotations

import copy
import logging
import os
import signal
import warnings
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

warnings.warn(
    "app.training.checkpointing is deprecated and scheduled for removal in Q2 2026. "
    "Use app.training.checkpoint_unified instead. "
    "See the module docstring for migration instructions.",
    DeprecationWarning,
    stacklevel=2,
)

# Use centralized executor pool (December 2025)
try:
    from app.coordination.async_bridge_manager import (
        get_bridge_manager,
        get_shared_executor,
    )
    HAS_BRIDGE_MANAGER = True
except ImportError:
    HAS_BRIDGE_MANAGER = False

import torch
import torch.nn as nn
import torch.optim as optim

from app.training.model_versioning import (
    LegacyCheckpointError,
    ModelVersionManager,
    VersionMismatchError,
)
from app.training.training_enhancements import EarlyStopping
from app.utils.resource_guard import LIMITS, check_disk_space, get_disk_usage
from app.utils.torch_utils import safe_load_checkpoint

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    scheduler: Any | None = None,
    early_stopping: EarlyStopping | None = None,
    use_versioning: bool = True,
) -> None:
    """
    Save a training checkpoint with optional versioning metadata.

    Args:
        model: The model to save
        optimizer: The optimizer to save state from
        epoch: Current epoch number
        loss: Current loss value
        path: Path to save checkpoint to
        scheduler: Optional LR scheduler to save state from
        early_stopping: Optional early stopping tracker to save state from
        use_versioning: Whether to include versioning metadata (default True)

    Raises:
        IOError: If disk space is insufficient (>70% used or <1GB available)
    """
    # Check disk space before saving (checkpoints can be 50-200MB)
    dir_path = os.path.dirname(path) if os.path.dirname(path) else '.'
    if not check_disk_space(required_gb=1.0, path=dir_path, log_warning=False):
        disk_pct, available_gb, _ = get_disk_usage(dir_path)
        raise OSError(
            f"Insufficient disk space to save checkpoint: "
            f"{disk_pct:.1f}% used (limit: {LIMITS.DISK_MAX_PERCENT}%), "
            f"{available_gb:.1f}GB available. Path: {path}"
        )

    # Ensure directory exists
    os.makedirs(dir_path, exist_ok=True)

    if use_versioning:
        # Use versioned checkpoint format
        manager = ModelVersionManager()
        training_info = {
            'epoch': epoch,
            'loss': float(loss),
        }
        if early_stopping is not None:
            training_info['early_stopping'] = {
                'best_loss': early_stopping.best_loss,
                'counter': early_stopping.counter,
            }

        metadata = manager.create_metadata(
            model,
            training_info=training_info,
        )

        manager.save_checkpoint(
            model,
            metadata,
            path,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            loss=loss,
        )

        # Also save early stopping best_state if needed for resume
        if (
            early_stopping is not None
            and early_stopping.best_state is not None
        ):
            # Save early stopping state separately so it survives reloading
            # Use atomic save pattern to prevent corruption
            es_path = Path(path.replace('.pth', '_early_stopping.pth'))
            es_temp_path = es_path.with_suffix('.pth.tmp')
            try:
                torch.save(
                    {
                        'best_loss': early_stopping.best_loss,
                        'counter': early_stopping.counter,
                        'best_state': early_stopping.best_state,
                    },
                    es_temp_path,
                )
                # Ensure flushed to disk before rename (December 2025)
                try:
                    with open(es_temp_path, 'rb') as f:
                        os.fsync(f.fileno())
                except OSError:
                    pass  # Best effort for early stopping state
                es_temp_path.rename(es_path)
            except Exception as e:
                es_temp_path.unlink(missing_ok=True)
                logger.warning(f"Failed to save early stopping state: {e}")

        logger.info(
            f"Saved versioned checkpoint to {path} "
            f"(epoch {epoch}, loss {loss:.4f}, "
            f"version {metadata.architecture_version})"
        )
    else:
        # Legacy format for backwards compatibility
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if early_stopping is not None:
            checkpoint['early_stopping'] = {
                'best_loss': early_stopping.best_loss,
                'counter': early_stopping.counter,
                'best_state': early_stopping.best_state,
            }

        # Use atomic save pattern to prevent corruption
        path_obj = Path(path)
        temp_path = path_obj.with_suffix('.pth.tmp')
        try:
            torch.save(checkpoint, temp_path)
            # Ensure checkpoint is flushed to disk before rename (December 2025)
            try:
                with open(temp_path, 'rb') as f:
                    os.fsync(f.fileno())
            except OSError:
                os.sync()  # Fallback for NFS/network filesystems
            temp_path.rename(path_obj)
        except Exception as e:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to save checkpoint: {e}")

        logger.info(
            "Saved legacy checkpoint to %s (epoch %d, loss %.4f)",
            path,
            epoch,
            loss,
        )


class AsyncCheckpointer:
    """
    Background checkpoint saver for non-blocking checkpoint I/O.

    Saves checkpoints in a background thread to avoid blocking the training loop.
    Provides 5-10% speedup by overlapping checkpoint I/O with GPU computation.

    Usage:
        checkpointer = AsyncCheckpointer(max_pending=2)

        # In training loop:
        checkpointer.save_async(model, optimizer, epoch, loss, path, ...)

        # At training end:
        checkpointer.wait_for_pending()
        checkpointer.shutdown()
    """

    def __init__(self, max_pending: int = 2, use_shared_executor: bool = True):
        """
        Initialize the async checkpointer.

        Args:
            max_pending: Maximum number of pending checkpoint saves.
                Older pending saves will be waited on before new ones start.
            use_shared_executor: If True and AsyncBridgeManager is available,
                use the shared executor pool. Otherwise create a private executor.
        """
        self._owns_executor = False
        self._use_shared = use_shared_executor and HAS_BRIDGE_MANAGER

        if self._use_shared:
            # Use centralized executor pool (December 2025)
            self._executor = get_shared_executor()
            # Register with bridge manager for lifecycle coordination
            try:
                manager = get_bridge_manager()
                manager.register_bridge(
                    "async_checkpointer",
                    self,
                    self._on_bridge_shutdown,
                )
            except Exception as e:
                logger.debug(f"Could not register with bridge manager: {e}")
        else:
            # Fallback to private executor
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint")
            self._owns_executor = True

        self._pending: deque = deque(maxlen=max_pending)
        self._max_pending = max_pending

    def _on_bridge_shutdown(self) -> None:
        """Callback when bridge manager is shutting down."""
        self.wait_for_pending()

    def save_async(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        loss: float,
        path: str,
        scheduler: Any | None = None,
        early_stopping: EarlyStopping | None = None,
        use_versioning: bool = True,
    ) -> None:
        """
        Queue a checkpoint for background saving.

        Makes a deep copy of model/optimizer state to avoid mutation during save.
        """
        # Wait for oldest pending save if at capacity
        if len(self._pending) >= self._max_pending:
            oldest_path, oldest_future = self._pending.popleft()
            try:
                oldest_future.result(timeout=120)
            except Exception as e:
                logger.error(f"Async checkpoint save failed for {oldest_path}: {e}")

        # Deep copy state dicts to prevent mutation during background save
        # Move tensors to CPU to reduce GPU memory and enable background copy
        model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        scheduler_state = copy.deepcopy(scheduler.state_dict()) if scheduler else None
        early_stopping_state = None
        if early_stopping is not None:
            early_stopping_state = {
                'best_loss': early_stopping.best_loss,
                'counter': early_stopping.counter,
                'best_state': {k: v.cpu().clone() for k, v in early_stopping.best_state.items()}
                if early_stopping.best_state else None,
            }

        # Submit to background thread
        future = self._executor.submit(
            self._save_worker,
            model_state,
            optimizer_state,
            epoch,
            loss,
            path,
            scheduler_state,
            early_stopping_state,
            use_versioning,
        )
        self._pending.append((path, future))
        logger.debug(f"Queued async checkpoint save: {path}")

    def _save_worker(
        self,
        model_state: dict,
        optimizer_state: dict,
        epoch: int,
        loss: float,
        path: str,
        scheduler_state: dict | None,
        early_stopping_state: dict | None,
        use_versioning: bool,
    ) -> None:
        """Background worker that performs the actual save."""
        dir_path = os.path.dirname(path) if os.path.dirname(path) else '.'
        os.makedirs(dir_path, exist_ok=True)

        # Build checkpoint dict
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'loss': loss,
        }
        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state
        if early_stopping_state is not None:
            checkpoint['early_stopping'] = early_stopping_state

        # Atomic save with temp file
        path_obj = Path(path)
        temp_path = path_obj.with_suffix('.pth.tmp')
        try:
            torch.save(checkpoint, temp_path)
            temp_path.rename(path_obj)
            logger.info(f"Async checkpoint saved: {path} (epoch {epoch}, loss {loss:.4f})")
        except Exception as e:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to save async checkpoint: {e}")

    def wait_for_pending(self, timeout: float = 120) -> None:
        """Wait for all pending checkpoint saves to complete."""
        for path, future in list(self._pending):
            try:
                future.result(timeout=timeout)
            except Exception as e:
                logger.error(f"Async checkpoint save failed for {path}: {e}")
        self._pending.clear()

    def shutdown(self) -> None:
        """Shutdown the executor and wait for pending saves."""
        self.wait_for_pending()
        # Only shutdown if we own the executor (not using shared pool)
        if self._owns_executor:
            self._executor.shutdown(wait=True)
        else:
            # Unregister from bridge manager
            try:
                manager = get_bridge_manager()
                manager.unregister_bridge("async_checkpointer")
            except Exception:
                pass


class GracefulShutdownHandler:
    """
    Handles graceful shutdown on SIGTERM/SIGINT signals.

    Saves an emergency checkpoint when the process receives a shutdown signal,
    preventing loss of training progress.

    Usage:
        handler = GracefulShutdownHandler()
        handler.setup(lambda: save_checkpoint(model, opt, epoch, loss, path))

        # Training loop...

        handler.teardown()
    """

    def __init__(self):
        self._shutdown_requested = False
        self._original_handlers: dict[int, Any] = {}
        self._checkpoint_callback: Callable[[], None] | None = None

    def setup(self, checkpoint_callback: Callable[[], None]) -> None:
        """
        Setup signal handlers for graceful shutdown.

        Args:
            checkpoint_callback: Function to call to save a checkpoint when shutdown is requested
        """
        self._checkpoint_callback = checkpoint_callback

        # Install signal handlers (only on main thread)
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._original_handlers[sig] = signal.signal(sig, self._handle_signal)
            logger.info("Graceful shutdown handlers installed")
        except ValueError:
            # Signal handling can only be set in main thread
            logger.debug("Signal handlers not installed (not main thread)")

    def teardown(self) -> None:
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass  # May fail if not in main thread
        self._original_handlers.clear()

    def _handle_signal(self, signum: int, frame) -> None:
        """Handle shutdown signal."""
        sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        logger.warning(f"Received {sig_name}, initiating graceful shutdown...")
        self._shutdown_requested = True

        # Save emergency checkpoint
        if self._checkpoint_callback:
            try:
                logger.info("Saving emergency checkpoint before shutdown...")
                self._checkpoint_callback()
                logger.info("Emergency checkpoint saved successfully")
            except Exception as e:
                logger.error(f"Failed to save emergency checkpoint: {e}")

        # Re-raise the signal to allow normal termination after checkpoint
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    scheduler: Any | None = None,
    early_stopping: EarlyStopping | None = None,
    device: torch.device | None = None,
    strict_versioning: bool = False,
) -> tuple[int, float]:
    """
    Load a training checkpoint with optional version validation.

    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional LR scheduler to load state into
        early_stopping: Optional early stopping tracker to restore state into
        device: Device to map checkpoint tensors to
        strict_versioning: If True, fail on version mismatch. If False,
            log warnings but continue (default: False for backwards compat)

    Returns:
        Tuple of (epoch, loss) from the checkpoint

    Raises:
        VersionMismatchError: If strict_versioning and version mismatch
    """
    # Use safe loading utility - tries weights_only=True first
    map_loc = str(device) if device else "cpu"
    checkpoint = safe_load_checkpoint(path, map_location=map_loc, warn_on_unsafe=False)

    # Check if this is a versioned checkpoint
    manager = ModelVersionManager(default_device=device)
    if manager.METADATA_KEY in checkpoint:
        # Versioned checkpoint
        try:
            state_dict, metadata = manager.load_checkpoint(
                path,
                strict=strict_versioning,
                verify_checksum=True,
                device=device,
            )
            model.load_state_dict(state_dict)
            logger.info(
                f"Loaded versioned checkpoint from {path} "
                f"(version {metadata.architecture_version})"
            )

            # Extract epoch/loss from metadata or checkpoint
            epoch = checkpoint.get('epoch', 0)
            loss = checkpoint.get('loss', float('inf'))

        except (VersionMismatchError, LegacyCheckpointError) as e:
            if strict_versioning:
                raise
            logger.warning(f"Version issue loading checkpoint: {e}")
            # Fall back to direct loading
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 0)
            loss = checkpoint.get('loss', float('inf'))
    else:
        # Legacy checkpoint format
        logger.info(f"Loading legacy checkpoint from {path}")
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load early stopping state
    if early_stopping is not None:
        if 'early_stopping' in checkpoint:
            es_state = checkpoint['early_stopping']
            early_stopping.best_loss = es_state['best_loss']
            early_stopping.counter = es_state['counter']
            early_stopping.best_state = es_state.get('best_state')
        else:
            # Check for separate early stopping file
            es_path = path.replace('.pth', '_early_stopping.pth')
            if os.path.exists(es_path):
                es_state = safe_load_checkpoint(es_path, map_location=device, warn_on_unsafe=False)
                early_stopping.best_loss = es_state['best_loss']
                early_stopping.counter = es_state['counter']
                early_stopping.best_state = es_state.get('best_state')

    logger.info(
        f"Loaded checkpoint from {path} (epoch {epoch}, loss {loss:.4f})"
    )
    return epoch, loss


def load_weights_only(
    path: str,
    model: nn.Module,
    device: torch.device | None = None,
    strict: bool = False,
) -> dict[str, list[str]]:
    """
    Load only model weights from a checkpoint (for transfer learning).

    Unlike load_checkpoint, this function:
    - Does NOT load optimizer/scheduler/early_stopping state
    - Allows partial loading when strict=False (skips mismatched layers)
    - Useful for transfer learning (e.g., 2-player model to 4-player)

    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        device: Device to map checkpoint tensors to
        strict: If True, require all keys to match. If False, skip mismatched layers.

    Returns:
        Dict with 'missing_keys' and 'unexpected_keys' lists

    Example:
        # Initialize 4-player model with 2-player weights
        result = load_weights_only(
            "models/canonical_sq8_2p.pth",
            model_4p,
            strict=False
        )
        print(f"Skipped layers: {result['missing_keys']}")
    """
    map_loc = str(device) if device else "cpu"
    checkpoint = safe_load_checkpoint(path, map_location=map_loc, warn_on_unsafe=False)

    # Extract state dict from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and any('conv' in k or 'res_blocks' in k for k in checkpoint.keys()):
        state_dict = checkpoint
    else:
        raise ValueError(f"Cannot find model weights in checkpoint: {path}")

    # Load with strict=False to allow partial loading
    result = model.load_state_dict(state_dict, strict=strict)

    if not strict:
        if result.missing_keys:
            logger.warning(
                f"Transfer learning: {len(result.missing_keys)} layers not in checkpoint "
                f"(initialized fresh): {result.missing_keys[:5]}..."
            )
        if result.unexpected_keys:
            logger.warning(
                f"Transfer learning: {len(result.unexpected_keys)} layers in checkpoint "
                f"but not in model (skipped): {result.unexpected_keys[:5]}..."
            )

    logger.info(
        f"Loaded weights from {path} (strict={strict}, "
        f"loaded={len(state_dict) - len(result.unexpected_keys)} layers)"
    )

    return {
        'missing_keys': list(result.missing_keys),
        'unexpected_keys': list(result.unexpected_keys),
    }


__all__ = [
    "AsyncCheckpointer",
    "GracefulShutdownHandler",
    "load_checkpoint",
    "load_weights_only",
    "save_checkpoint",
]
