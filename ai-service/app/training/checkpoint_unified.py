"""Unified Checkpoint Manager for RingRift AI.

Consolidates checkpoint management from:
- fault_tolerance.py: Comprehensive metadata, hash verification, checkpoint types
- advanced_training.py SmartCheckpointManager: Adaptive frequency based on loss improvement

This provides a single checkpoint manager with all advanced features.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Use shared lazy torch import to prevent OOM
from app.training.utils import get_torch

# Event emission for checkpoint observability (optional)
try:
    from app.distributed.data_events import emit_checkpoint_saved, emit_checkpoint_loaded
    HAS_CHECKPOINT_EVENTS = True
except ImportError:
    HAS_CHECKPOINT_EVENTS = False
    emit_checkpoint_saved = None
    emit_checkpoint_loaded = None


class CheckpointType(Enum):
    """Types of checkpoints."""
    REGULAR = "regular"          # Periodic checkpoint
    EPOCH = "epoch"              # End of epoch
    BEST = "best"                # Best performance
    EMERGENCY = "emergency"      # Before potential failure
    RECOVERY = "recovery"        # After recovery


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint with rich tracking information."""
    checkpoint_id: str
    checkpoint_type: CheckpointType
    epoch: int
    global_step: int
    timestamp: datetime
    metrics: Dict[str, float]
    training_config: Dict[str, Any]
    file_path: str
    file_hash: str
    parent_checkpoint: Optional[str] = None
    # Additional metadata from SmartCheckpointManager
    adaptive_interval: Optional[int] = None
    improvement_rate: Optional[float] = None
    # Architecture versioning metadata (integrated from model_versioning.py)
    architecture_version: Optional[str] = None
    model_class: Optional[str] = None
    model_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['checkpoint_type'] = self.checkpoint_type.value
        d['timestamp'] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CheckpointMetadata':
        d = d.copy()
        d['checkpoint_type'] = CheckpointType(d['checkpoint_type'])
        d['timestamp'] = datetime.fromisoformat(d['timestamp'])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingProgress:
    """Tracks training progress for recovery and resumption."""
    epoch: int = 0
    global_step: int = 0
    batch_idx: int = 0
    samples_seen: int = 0
    best_metric: Optional[float] = None
    best_metric_name: str = "loss"
    best_epoch: int = 0
    total_epochs: int = 100
    learning_rate: float = 0.001
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    random_state: Optional[Dict[str, Any]] = None
    extra_state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingProgress':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class UnifiedCheckpointConfig:
    """Configuration for unified checkpoint manager."""
    # Basic settings
    checkpoint_dir: Path = Path("checkpoints")

    # Retention policy (from fault_tolerance)
    max_checkpoints: int = 10
    keep_best: int = 3
    keep_every_n_epochs: int = 10

    # Adaptive checkpointing (from SmartCheckpointManager)
    adaptive_enabled: bool = True
    min_interval_epochs: int = 1
    max_interval_epochs: int = 10
    improvement_threshold: float = 0.01

    # Step-based checkpointing
    checkpoint_interval_steps: int = 1000

    # Verification
    verify_hash: bool = True

    # Async saving
    async_save: bool = False


class UnifiedCheckpointManager:
    """
    Unified checkpoint manager combining comprehensive metadata tracking
    with adaptive checkpointing frequency.

    Features:
    - Checkpoint types: REGULAR, EPOCH, BEST, EMERGENCY, RECOVERY
    - SHA256 hash verification for integrity
    - Parent checkpoint tracking (checkpoint lineage)
    - Adaptive checkpointing based on loss improvement
    - Flexible retention policy
    - Thread-safe async saving option

    Usage:
        manager = UnifiedCheckpointManager(UnifiedCheckpointConfig(
            checkpoint_dir=Path("checkpoints"),
            keep_best=3,
            adaptive_enabled=True,
        ))

        # Check if should save with adaptive logic
        if manager.should_save(epoch=5, loss=0.25):
            manager.save_checkpoint(
                model_state=model.state_dict(),
                progress=progress,
                metrics={"loss": 0.25},
            )
    """

    def __init__(self, config: Optional[UnifiedCheckpointConfig] = None):
        self.config = config or UnifiedCheckpointConfig()
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metadata tracking
        self.metadata_file = self.checkpoint_dir / "checkpoints.json"
        self.checkpoints: List[CheckpointMetadata] = []

        # Adaptive checkpointing state (from SmartCheckpointManager)
        self._last_save_epoch = -1
        self._last_save_step = -1
        self._last_loss = float('inf')
        self._adaptive_interval = self.config.min_interval_epochs

        # Async saving
        self._save_lock = threading.Lock()
        self._save_thread: Optional[threading.Thread] = None

        self._load_metadata()

    def _load_metadata(self):
        """Load checkpoint metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    data = json.load(f)
                self.checkpoints = [
                    CheckpointMetadata.from_dict(c) for c in data.get('checkpoints', [])
                ]
                # Restore adaptive state
                adaptive_state = data.get('adaptive_state', {})
                self._last_save_epoch = adaptive_state.get('last_save_epoch', -1)
                self._last_save_step = adaptive_state.get('last_save_step', -1)
                self._last_loss = adaptive_state.get('last_loss', float('inf'))
                self._adaptive_interval = adaptive_state.get('adaptive_interval',
                                                             self.config.min_interval_epochs)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load checkpoint metadata: {e}")
                self.checkpoints = []

    def _save_metadata(self):
        """Save checkpoint metadata to disk."""
        data = {
            'checkpoints': [c.to_dict() for c in self.checkpoints],
            'adaptive_state': {
                'last_save_epoch': self._last_save_epoch,
                'last_save_step': self._last_save_step,
                'last_loss': self._last_loss,
                'adaptive_interval': self._adaptive_interval,
            },
            'updated_at': datetime.now().isoformat()
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of checkpoint file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]

    def _generate_checkpoint_id(
        self,
        epoch: int,
        step: int,
        checkpoint_type: CheckpointType
    ) -> str:
        """Generate unique checkpoint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ckpt_e{epoch:04d}_s{step:08d}_{checkpoint_type.value}_{timestamp}"

    def should_save(
        self,
        epoch: int,
        loss: float,
        step: Optional[int] = None,
        force: bool = False,
    ) -> bool:
        """
        Determine if checkpoint should be saved using adaptive logic.

        Uses improvement-based adaptive interval:
        - Always save if significant improvement detected
        - Otherwise, use adaptive interval that increases when training plateaus

        Args:
            epoch: Current epoch
            loss: Current loss value
            step: Current step (optional, for step-based checkpointing)
            force: Force checkpoint regardless of adaptive logic

        Returns:
            True if checkpoint should be saved
        """
        if force:
            return True

        if not self.config.adaptive_enabled:
            # Fall back to simple interval
            if epoch - self._last_save_epoch >= self.config.min_interval_epochs:
                return True
            if step is not None and step - self._last_save_step >= self.config.checkpoint_interval_steps:
                return True
            return False

        # Adaptive logic from SmartCheckpointManager
        if epoch - self._last_save_epoch < self.config.min_interval_epochs:
            return False

        # Calculate improvement
        if self._last_loss == float('inf'):
            improvement = 1.0  # First checkpoint
        else:
            improvement = (self._last_loss - loss) / (abs(self._last_loss) + 1e-8)

        # Always save if significant improvement
        if improvement > self.config.improvement_threshold:
            self._adaptive_interval = self.config.min_interval_epochs
            return True

        # Adaptive interval based on improvement rate
        if epoch - self._last_save_epoch >= self._adaptive_interval:
            # Increase interval when plateauing (save less frequently)
            self._adaptive_interval = min(
                self._adaptive_interval + 1,
                self.config.max_interval_epochs
            )
            return True

        return False

    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        progress: TrainingProgress,
        checkpoint_type: CheckpointType = CheckpointType.REGULAR,
        metrics: Optional[Dict[str, float]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        optimizer_state: Optional[Dict[str, Any]] = None,
        scheduler_state: Optional[Dict[str, Any]] = None,
        architecture_version: Optional[str] = None,
        model_class: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> CheckpointMetadata:
        """
        Save a training checkpoint with full metadata.

        Args:
            model_state: Model state dict
            progress: Training progress state
            checkpoint_type: Type of checkpoint
            metrics: Current training metrics
            training_config: Training configuration
            optimizer_state: Optional optimizer state
            scheduler_state: Optional scheduler state
            architecture_version: Model architecture version (e.g., "v2.0.0")
            model_class: Model class name (e.g., "RingRiftCNN_v2")
            model_config: Model configuration dict for compatibility validation

        Returns:
            CheckpointMetadata for the saved checkpoint
        """
        torch = get_torch()

        metrics = metrics or {}

        def _do_save():
            nonlocal metrics

            with self._save_lock:
                checkpoint_id = self._generate_checkpoint_id(
                    progress.epoch, progress.global_step, checkpoint_type
                )
                file_path = self.checkpoint_dir / f"{checkpoint_id}.pt"

                # Build checkpoint data
                checkpoint_data = {
                    'model_state_dict': model_state,
                    'progress': progress.to_dict(),
                    'checkpoint_type': checkpoint_type.value,
                    'metrics': metrics,
                    'training_config': training_config or {},
                    'timestamp': datetime.now().isoformat(),
                    'adaptive_state': {
                        'interval': self._adaptive_interval,
                        'improvement_rate': self._calculate_improvement_rate(metrics.get('loss')),
                    }
                }

                if optimizer_state is not None:
                    checkpoint_data['optimizer_state_dict'] = optimizer_state
                if scheduler_state is not None:
                    checkpoint_data['scheduler_state_dict'] = scheduler_state

                # Add architecture versioning metadata (compatible with model_versioning.py)
                if architecture_version or model_class or model_config:
                    checkpoint_data['_versioning_metadata'] = {
                        'architecture_version': architecture_version or 'unknown',
                        'model_class': model_class or 'Unknown',
                        'config': model_config or {},
                        'created_at': datetime.now().isoformat(),
                    }

                # Save checkpoint
                torch.save(checkpoint_data, file_path)

                # Ensure checkpoint is flushed to disk before continuing (December 2025)
                # This prevents emitting CHECKPOINT_SAVED before data is durable
                try:
                    with open(file_path, 'rb') as f:
                        os.fsync(f.fileno())
                except (OSError, IOError) as e:
                    # fsync may fail on some filesystems (e.g., NFS without sync option)
                    # Fall back to global sync as a last resort
                    logger.debug(f"fsync failed ({e}), using os.sync() fallback")
                    os.sync()

                # Compute hash if enabled
                file_hash = self._compute_hash(file_path) if self.config.verify_hash else ""

                # Create metadata
                parent = self.checkpoints[-1].checkpoint_id if self.checkpoints else None
                metadata = CheckpointMetadata(
                    checkpoint_id=checkpoint_id,
                    checkpoint_type=checkpoint_type,
                    epoch=progress.epoch,
                    global_step=progress.global_step,
                    timestamp=datetime.now(),
                    metrics=metrics,
                    training_config=training_config or {},
                    file_path=str(file_path),
                    file_hash=file_hash,
                    parent_checkpoint=parent,
                    adaptive_interval=self._adaptive_interval,
                    improvement_rate=self._calculate_improvement_rate(metrics.get('loss')),
                    architecture_version=architecture_version,
                    model_class=model_class,
                    model_config=model_config,
                )

                self.checkpoints.append(metadata)

                # Update adaptive state
                self._last_save_epoch = progress.epoch
                self._last_save_step = progress.global_step
                if 'loss' in metrics:
                    self._last_loss = metrics['loss']

                self._save_metadata()

                # Cleanup old checkpoints
                self._cleanup_checkpoints(metrics)

                logger.info(f"Saved checkpoint: {checkpoint_id} (adaptive_interval={self._adaptive_interval})")

                # Emit checkpoint event for observability
                if HAS_CHECKPOINT_EVENTS and emit_checkpoint_saved is not None:
                    try:
                        import asyncio
                        asyncio.get_event_loop().create_task(emit_checkpoint_saved(
                            config=self.config.board_type if hasattr(self.config, 'board_type') else "",
                            checkpoint_path=str(file_path),
                            epoch=progress.epoch,
                            step=progress.global_step,
                            metrics=metrics,
                            source="checkpoint_unified",
                        ))
                    except RuntimeError:
                        pass  # No running event loop - skip event emission

                return metadata

        if self.config.async_save:
            # Run save in background thread
            self._save_thread = threading.Thread(target=_do_save, daemon=True)
            self._save_thread.start()
            # Return a placeholder metadata
            return CheckpointMetadata(
                checkpoint_id="pending",
                checkpoint_type=checkpoint_type,
                epoch=progress.epoch,
                global_step=progress.global_step,
                timestamp=datetime.now(),
                metrics=metrics,
                training_config=training_config or {},
                file_path="",
                file_hash="",
            )
        else:
            return _do_save()

    def _calculate_improvement_rate(self, current_loss: Optional[float]) -> Optional[float]:
        """Calculate improvement rate from last checkpoint."""
        if current_loss is None or self._last_loss == float('inf'):
            return None
        return (self._last_loss - current_loss) / (abs(self._last_loss) + 1e-8)

    def _cleanup_checkpoints(self, current_metrics: Optional[Dict[str, float]] = None):
        """Remove old checkpoints based on retention policy."""
        if len(self.checkpoints) <= self.config.max_checkpoints:
            return

        # Identify checkpoints to keep
        keep_ids = set()

        # Keep best checkpoints by loss
        if current_metrics or any(c.metrics.get('loss') is not None for c in self.checkpoints):
            sorted_by_loss = sorted(
                [c for c in self.checkpoints if c.metrics.get('loss') is not None],
                key=lambda c: c.metrics.get('loss', float('inf'))
            )
            for c in sorted_by_loss[:self.config.keep_best]:
                keep_ids.add(c.checkpoint_id)

        # Keep epoch checkpoints (for long-term recovery)
        for c in self.checkpoints:
            if c.epoch % self.config.keep_every_n_epochs == 0:
                keep_ids.add(c.checkpoint_id)

        # Keep most recent checkpoints
        for c in self.checkpoints[-self.config.max_checkpoints:]:
            keep_ids.add(c.checkpoint_id)

        # Always keep emergency and best type checkpoints
        for c in self.checkpoints:
            if c.checkpoint_type in (CheckpointType.EMERGENCY, CheckpointType.BEST):
                keep_ids.add(c.checkpoint_id)

        # Remove checkpoints not in keep set
        to_remove = [c for c in self.checkpoints if c.checkpoint_id not in keep_ids]
        for c in to_remove:
            try:
                Path(c.file_path).unlink()
                logger.debug(f"Removed old checkpoint: {c.checkpoint_id}")
            except OSError:
                pass
            self.checkpoints.remove(c)

        self._save_metadata()

    def load_checkpoint(
        self,
        checkpoint_id: Optional[str] = None,
        checkpoint_type: Optional[CheckpointType] = None,
        best_by_metric: Optional[str] = None,
        device: str = 'cpu',
        expected_version: Optional[str] = None,
        expected_class: Optional[str] = None,
        strict_version: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.

        Args:
            checkpoint_id: Specific checkpoint to load
            checkpoint_type: Load latest checkpoint of this type
            best_by_metric: Load best checkpoint by this metric (e.g., 'loss')
            device: Device to load checkpoint to
            expected_version: Expected architecture version for validation
            expected_class: Expected model class name for validation
            strict_version: If True, raise error on version mismatch; if False, log warning

        Returns:
            Checkpoint data dict or None

        Raises:
            ValueError: If strict_version=True and version/class mismatch detected
        """
        torch = get_torch()

        metadata = None

        if checkpoint_id:
            for c in self.checkpoints:
                if c.checkpoint_id == checkpoint_id:
                    metadata = c
                    break
        elif best_by_metric:
            valid = [c for c in self.checkpoints if best_by_metric in c.metrics]
            if valid:
                metadata = min(valid, key=lambda c: c.metrics[best_by_metric])
        elif checkpoint_type:
            for c in reversed(self.checkpoints):
                if c.checkpoint_type == checkpoint_type:
                    metadata = c
                    break
        else:
            # Load most recent
            if self.checkpoints:
                metadata = self.checkpoints[-1]

        if not metadata:
            return None

        file_path = Path(metadata.file_path)
        if not file_path.exists():
            logger.error(f"Checkpoint file not found: {file_path}")
            return None

        # Verify hash if enabled
        if self.config.verify_hash and metadata.file_hash:
            computed_hash = self._compute_hash(file_path)
            if computed_hash != metadata.file_hash:
                logger.warning(f"Checkpoint hash mismatch for {metadata.checkpoint_id}")

        checkpoint_data = torch.load(file_path, map_location=device)
        checkpoint_data['metadata'] = metadata

        # Validate architecture version if requested
        versioning_meta = checkpoint_data.get('_versioning_metadata', {})
        if expected_version or expected_class:
            ckpt_version = versioning_meta.get('architecture_version')
            ckpt_class = versioning_meta.get('model_class')

            version_mismatch = expected_version and ckpt_version and ckpt_version != expected_version
            class_mismatch = expected_class and ckpt_class and ckpt_class != expected_class

            if version_mismatch or class_mismatch:
                msg = f"Architecture mismatch for checkpoint {metadata.checkpoint_id}:"
                if version_mismatch:
                    msg += f" version={ckpt_version} (expected {expected_version})"
                if class_mismatch:
                    msg += f" class={ckpt_class} (expected {expected_class})"

                if strict_version:
                    raise ValueError(msg)
                else:
                    logger.warning(msg)

        # Emit checkpoint loaded event for observability
        if HAS_CHECKPOINT_EVENTS and emit_checkpoint_loaded is not None:
            try:
                import asyncio
                progress = checkpoint_data.get('progress', {})
                asyncio.get_event_loop().create_task(emit_checkpoint_loaded(
                    config=self.config.board_type if hasattr(self.config, 'board_type') else "",
                    checkpoint_path=str(file_path),
                    epoch=progress.get('epoch', 0),
                    step=progress.get('global_step', 0),
                    source="checkpoint_unified",
                ))
            except RuntimeError:
                pass  # No running event loop - skip event emission

        return checkpoint_data

    def get_best_checkpoint(
        self,
        metric_name: str = 'loss',
        lower_is_better: bool = True
    ) -> Optional[CheckpointMetadata]:
        """Get the best checkpoint by a metric."""
        valid = [c for c in self.checkpoints if metric_name in c.metrics]
        if not valid:
            return None

        return min(valid, key=lambda c: c.metrics[metric_name] * (1 if lower_is_better else -1))

    def get_latest_checkpoint(self) -> Optional[CheckpointMetadata]:
        """Get the most recent checkpoint."""
        return self.checkpoints[-1] if self.checkpoints else None

    def list_checkpoints(
        self,
        checkpoint_type: Optional[CheckpointType] = None
    ) -> List[CheckpointMetadata]:
        """List checkpoints, optionally filtered by type."""
        if checkpoint_type:
            return [c for c in self.checkpoints if c.checkpoint_type == checkpoint_type]
        return list(self.checkpoints)

    def get_stats(self) -> Dict[str, Any]:
        """Get checkpoint manager statistics."""
        return {
            'num_checkpoints': len(self.checkpoints),
            'best_loss': min((c.metrics.get('loss', float('inf')) for c in self.checkpoints), default=None),
            'adaptive_interval': self._adaptive_interval,
            'last_save_epoch': self._last_save_epoch,
            'last_save_step': self._last_save_step,
            'checkpoint_types': {
                t.value: len([c for c in self.checkpoints if c.checkpoint_type == t])
                for t in CheckpointType
            }
        }

    def save_best_if_improved(
        self,
        model_state: Dict[str, Any],
        progress: TrainingProgress,
        metric_name: str,
        metric_value: float,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[CheckpointMetadata]:
        """
        Save a BEST checkpoint only if this is an improvement.

        Args:
            model_state: Model state dict
            progress: Training progress
            metric_name: Name of metric to compare
            metric_value: Current metric value
            training_config: Training configuration

        Returns:
            CheckpointMetadata if saved, None otherwise
        """
        # Check if this is the best
        current_best = self.get_best_checkpoint(metric_name)

        if current_best is not None:
            if metric_value >= current_best.metrics.get(metric_name, float('inf')):
                return None  # Not an improvement

        # Save as best
        return self.save_checkpoint(
            model_state=model_state,
            progress=progress,
            checkpoint_type=CheckpointType.BEST,
            metrics={metric_name: metric_value},
            training_config=training_config,
        )

    def wait_for_async_save(self, timeout: float = 30.0):
        """Wait for async save to complete."""
        if self._save_thread is not None and self._save_thread.is_alive():
            self._save_thread.join(timeout=timeout)


# Backwards compatibility aliases
SmartCheckpointManager = UnifiedCheckpointManager
CheckpointManager = UnifiedCheckpointManager

# Re-export legacy checkpointing functions for migration
# This allows callers to import from checkpoint_unified instead of deprecated checkpointing
try:
    from app.training.checkpointing import (
        GracefulShutdownHandler,
        save_checkpoint,
        load_checkpoint,
        AsyncCheckpointer,
    )
    _HAS_LEGACY_CHECKPOINTING = True
except ImportError:
    _HAS_LEGACY_CHECKPOINTING = False
    GracefulShutdownHandler = None  # type: ignore
    save_checkpoint = None  # type: ignore
    load_checkpoint = None  # type: ignore
    AsyncCheckpointer = None  # type: ignore


def create_checkpoint_manager(
    checkpoint_dir: Union[str, Path],
    keep_best: int = 3,
    adaptive_enabled: bool = True,
    **kwargs,
) -> UnifiedCheckpointManager:
    """Factory function to create checkpoint manager.

    Args:
        checkpoint_dir: Directory for checkpoints
        keep_best: Number of best checkpoints to keep
        adaptive_enabled: Enable adaptive checkpointing frequency
        **kwargs: Additional config options

    Returns:
        Configured UnifiedCheckpointManager
    """
    config = UnifiedCheckpointConfig(
        checkpoint_dir=Path(checkpoint_dir),
        keep_best=keep_best,
        adaptive_enabled=adaptive_enabled,
        **{k: v for k, v in kwargs.items() if hasattr(UnifiedCheckpointConfig, k)},
    )
    return UnifiedCheckpointManager(config)
