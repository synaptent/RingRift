"""
Training Fault Tolerance for RingRift AI.

Provides checkpointing, recovery, and fault handling for robust training.
"""

import json
import logging
import os
import signal
import threading
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TypeVar

from app.utils.checksum_utils import compute_file_checksum
from app.utils.torch_utils import safe_load_checkpoint

logger = logging.getLogger(__name__)

# Re-export checkpoint classes from checkpoint_unified.py (canonical source)
# All checkpoint management is consolidated in checkpoint_unified.py
try:
    from app.training.checkpoint_unified import (
        UnifiedCheckpointConfig,
        UnifiedCheckpointManager,
        create_checkpoint_manager,
    )
    _HAS_UNIFIED_CHECKPOINT = True
except ImportError:
    _HAS_UNIFIED_CHECKPOINT = False
    UnifiedCheckpointManager = None  # type: ignore
    UnifiedCheckpointConfig = None  # type: ignore
    create_checkpoint_manager = None  # type: ignore
    # Fallback definitions below will be used

T = TypeVar('T')


# =============================================================================
# Retry Decorator with Exponential Backoff
# =============================================================================
# NOTE: December 2025 - Consolidated to use app.core.error_handler as canonical source
# The implementations below are backward-compatible wrappers

# Import canonical retry decorators from error_handler.py
try:
    from app.core.error_handler import (
        retry as _canonical_retry,
        retry_async as _canonical_retry_async,
    )
    _HAS_CANONICAL_RETRY = True
except ImportError:
    _HAS_CANONICAL_RETRY = False
    _canonical_retry = None
    _canonical_retry_async = None


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Callable[[Exception, int, float], None] | None = None,
    jitter: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that retries a function with exponential backoff.

    .. deprecated:: December 2025
        For new code, prefer using::

            from app.core.error_handler import retry

        Or for training-specific operations::

            from app.training.exception_integration import (
                retry_checkpoint_save,
                retry_training_step,
                TrainingRetryPolicies,
            )

        This wrapper is maintained for backward compatibility only.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback(exception, attempt, delay) called before each retry
        jitter: Add random jitter to prevent thundering herd (default True)

    Returns:
        Decorated function with retry logic
    """
    import random
    import warnings
    warnings.warn(
        "retry_with_backoff is deprecated. Use app.core.error_handler.retry "
        "or app.training.exception_integration.TrainingRetryPolicies instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        raise

                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    if on_retry:
                        on_retry(e, attempt + 1, delay)

                    time.sleep(delay)

            # Should never reach here, but for type safety
            raise last_exception  # type: ignore

        return wrapper
    return decorator


def async_retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Callable[[Exception, int, float], None] | None = None,
    jitter: bool = True,
    circuit_breaker_key: str | None = None,
):
    """
    Async decorator that retries an async function with exponential backoff.

    .. deprecated:: December 2025
        For new code, prefer using::

            from app.core.error_handler import retry_async

        This wrapper is maintained for backward compatibility only.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback(exception, attempt, delay) called before each retry
        jitter: Add random jitter to prevent thundering herd (default True)
        circuit_breaker_key: Optional key for circuit breaker integration

    Returns:
        Decorated async function with retry logic
    """
    import asyncio
    import random
    import warnings
    warnings.warn(
        "async_retry_with_backoff is deprecated. Use app.core.error_handler.retry_async instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Try to import circuit breaker
    try:
        from app.distributed.circuit_breaker import get_operation_breaker
        HAS_CIRCUIT_BREAKER = True
    except ImportError:
        HAS_CIRCUIT_BREAKER = False
        get_operation_breaker = None

    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            breaker = None

            # Get circuit breaker if configured
            if HAS_CIRCUIT_BREAKER and circuit_breaker_key:
                breaker = get_operation_breaker(circuit_breaker_key)

            for attempt in range(max_retries + 1):
                # Check circuit breaker before attempt
                if breaker:
                    host = kwargs.get('host', args[0] if args else 'default')
                    if not breaker.can_execute(str(host)):
                        raise RecoverableError(f"Circuit breaker open for {host}")

                try:
                    result = await func(*args, **kwargs)
                    # Record success in circuit breaker
                    if breaker:
                        host = kwargs.get('host', args[0] if args else 'default')
                        breaker.record_success(str(host))
                    return result

                except exceptions as e:
                    last_exception = e

                    # Record failure in circuit breaker
                    if breaker:
                        host = kwargs.get('host', args[0] if args else 'default')
                        breaker.record_failure(str(host), e)

                    if attempt == max_retries:
                        raise

                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"Async attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    if on_retry:
                        on_retry(e, attempt + 1, delay)

                    await asyncio.sleep(delay)

            # Should never reach here, but for type safety
            raise last_exception  # type: ignore

        return wrapper
    return decorator


class RetryPolicy:
    """Configurable retry policy for different operation types.

    Provides pre-configured policies for common scenarios:
    - AGGRESSIVE: Fast retries for time-sensitive operations
    - STANDARD: Default balanced retry policy
    - CONSERVATIVE: Slow retries for heavy operations
    - NETWORK: Network-optimized with longer delays

    NOTE (December 2025): For training-specific operations, prefer using:
        from app.training.exception_integration import TrainingRetryPolicies

    TrainingRetryPolicies provides specialized policies for:
    - CHECKPOINT_SAVE/LOAD: Optimized for I/O operations
    - TRAINING_STEP: Quick retries for GPU errors
    - EVALUATION/SELFPLAY: Longer delays for game-playing
    - PROMOTION/REMOTE_SYNC: Critical operations
    """

    # Pre-configured policies
    AGGRESSIVE = {
        "max_retries": 5,
        "base_delay": 0.5,
        "max_delay": 10.0,
        "exponential_base": 1.5,
    }

    STANDARD = {
        "max_retries": 3,
        "base_delay": 1.0,
        "max_delay": 60.0,
        "exponential_base": 2.0,
    }

    CONSERVATIVE = {
        "max_retries": 3,
        "base_delay": 5.0,
        "max_delay": 300.0,
        "exponential_base": 2.0,
    }

    NETWORK = {
        "max_retries": 4,
        "base_delay": 2.0,
        "max_delay": 120.0,
        "exponential_base": 2.0,
    }

    @classmethod
    def get_policy(cls, name: str) -> dict[str, Any]:
        """Get a pre-configured retry policy by name."""
        policies = {
            "aggressive": cls.AGGRESSIVE,
            "standard": cls.STANDARD,
            "conservative": cls.CONSERVATIVE,
            "network": cls.NETWORK,
        }
        return policies.get(name.lower(), cls.STANDARD)

    @staticmethod
    def apply(policy: dict[str, Any], sync: bool = True):
        """Apply a retry policy as a decorator.

        Args:
            policy: Dict with retry configuration
            sync: If True, returns sync decorator; if False, returns async decorator

        Usage:
            @RetryPolicy.apply(RetryPolicy.NETWORK, sync=False)
            async def fetch_remote_data(url):
                ...
        """
        if sync:
            return retry_with_backoff(**policy)
        else:
            return async_retry_with_backoff(**policy)


# =============================================================================
# Training Exception Hierarchy - Imported from canonical source
# =============================================================================

# Import all training-related errors from the unified errors module
# These are re-exported for backwards compatibility
try:
    from app.errors import (
        CheckpointError,
        DataQualityError,
        LifecycleError,
        NonRecoverableError,
        RecoverableError,
        ResourceError,
        TrainingError,
        ValidationError,
    )
except ImportError:
    # Fallback definitions for when app.errors is not available
    class TrainingError(Exception):
        """Base exception for all training-related errors."""
        def __init__(self, message: str, context: dict[str, Any] | None = None):
            super().__init__(message)
            self.context = context or {}

    class RecoverableError(TrainingError):
        """Exception that indicates a recoverable error (can retry)."""

    class NonRecoverableError(TrainingError):
        """Exception that indicates a non-recoverable error (skip retry)."""

    class ValidationError(TrainingError):
        """Exception for configuration or data validation failures."""

    class DataQualityError(TrainingError):
        """Exception for data quality issues."""

    class LifecycleError(TrainingError):
        """Exception for model lifecycle errors."""

    class ResourceError(TrainingError):
        """Exception for resource-related failures."""

    class CheckpointError(TrainingError):
        """Exception for checkpoint-related failures."""


def handle_gpu_error(
    fallback_to_cpu: bool = True,
    clear_cache: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that handles GPU-related errors with optional CPU fallback.

    Args:
        fallback_to_cpu: If True, retry on CPU after GPU failure
        clear_cache: If True, clear GPU cache after OOM errors

    Usage:
        @handle_gpu_error(fallback_to_cpu=True)
        def forward_pass(model, inputs, device):
            return model(inputs.to(device))
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                error_msg = str(e).lower()

                # Check for CUDA/MPS OOM errors
                if "out of memory" in error_msg or "cuda" in error_msg or "mps" in error_msg:
                    logger.warning(f"GPU error detected: {e}")

                    if clear_cache:
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                                # MPS doesn't have direct cache clearing, but we can gc
                                import gc
                                gc.collect()
                            logger.info("Cleared GPU cache")
                        except Exception as cache_err:
                            logger.warning(f"Failed to clear GPU cache: {cache_err}")

                    if fallback_to_cpu:
                        logger.info("Falling back to CPU")
                        # Try to modify device in kwargs if present
                        if 'device' in kwargs:
                            import torch
                            kwargs['device'] = torch.device('cpu')
                        return func(*args, **kwargs)

                raise

        return wrapper
    return decorator


class TrainingErrorHandler:
    """
    Centralized error handling for training with configurable strategies.

    Provides:
    - OOM handling with batch size reduction
    - Checkpoint recovery on failure
    - GPU failure fallback
    - Configurable retry strategies

    Usage:
        handler = TrainingErrorHandler(
            checkpoint_manager=ckpt_manager,
            min_batch_size=8,
        )

        with handler.safe_training_step(batch_size=256) as ctx:
            loss = train_step(batch)
            ctx.record_success()

        # Access recommended batch size after OOM
        new_batch_size = handler.recommended_batch_size
    """

    def __init__(
        self,
        checkpoint_manager: Optional['CheckpointManager'] = None,
        max_retries: int = 3,
        min_batch_size: int = 8,
        batch_reduction_factor: float = 0.5,
    ):
        self.checkpoint_manager = checkpoint_manager
        self.max_retries = max_retries
        self.min_batch_size = min_batch_size
        self.batch_reduction_factor = batch_reduction_factor

        self._current_batch_size: int | None = None
        self._consecutive_failures = 0
        self._oom_count = 0
        self._total_recoveries = 0

    @property
    def recommended_batch_size(self) -> int | None:
        """Get recommended batch size after OOM events."""
        return self._current_batch_size

    def reset_failure_count(self):
        """Reset consecutive failure counter (call after successful step)."""
        self._consecutive_failures = 0

    def handle_oom(self, current_batch_size: int) -> int:
        """
        Handle OOM error and return reduced batch size.

        Args:
            current_batch_size: Current batch size that caused OOM

        Returns:
            Reduced batch size to try
        """
        self._oom_count += 1

        new_size = max(
            self.min_batch_size,
            int(current_batch_size * self.batch_reduction_factor)
        )

        if new_size == current_batch_size:
            raise NonRecoverableError(
                f"Cannot reduce batch size below minimum ({self.min_batch_size})"
            )

        logger.warning(
            f"OOM detected. Reducing batch size: {current_batch_size} -> {new_size}"
        )
        self._current_batch_size = new_size
        return new_size

    @contextmanager
    def safe_training_step(self, batch_size: int):
        """
        Context manager for safe training step execution.

        Handles OOM, checkpointing, and recovery.

        Args:
            batch_size: Current batch size

        Yields:
            Context object with record_success() method
        """
        self._current_batch_size = batch_size

        class StepContext:
            def __init__(ctx_self):
                ctx_self.success = False

            def record_success(ctx_self):
                ctx_self.success = True
                self.reset_failure_count()
                self._total_recoveries += 1 if self._consecutive_failures > 0 else 0

        ctx = StepContext()

        try:
            yield ctx
        except RuntimeError as e:
            error_msg = str(e).lower()
            self._consecutive_failures += 1

            if "out of memory" in error_msg:
                # OOM - reduce batch size
                self.handle_oom(batch_size)
                raise RecoverableError(f"OOM with batch_size={batch_size}") from e

            elif self._consecutive_failures > self.max_retries:
                # Too many failures - save emergency checkpoint
                if self.checkpoint_manager:
                    logger.error("Max retries exceeded, saving emergency checkpoint")
                raise NonRecoverableError(
                    f"Max retries ({self.max_retries}) exceeded"
                ) from e

            else:
                raise RecoverableError(str(e)) from e

    def get_stats(self) -> dict[str, Any]:
        """Get error handling statistics."""
        return {
            "oom_count": self._oom_count,
            "consecutive_failures": self._consecutive_failures,
            "total_recoveries": self._total_recoveries,
            "current_batch_size": self._current_batch_size,
        }


# Only define CheckpointType if not imported from checkpoint_unified
if _HAS_UNIFIED_CHECKPOINT:
    from app.training.checkpoint_unified import CheckpointType
else:
    class CheckpointType(Enum):
        """Types of checkpoints."""
        REGULAR = "regular"          # Periodic checkpoint
        EPOCH = "epoch"              # End of epoch
        BEST = "best"                # Best performance
        EMERGENCY = "emergency"      # Before potential failure
        RECOVERY = "recovery"        # After recovery


class TrainingState(Enum):
    """Training process states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    CHECKPOINTING = "checkpointing"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStage(Enum):
    """Unified training pipeline stages (December 2025).

    The training pipeline follows this state machine:

        IDLE → SELFPLAY → DATA_SYNC → TRAINING → EVALUATION → PROMOTION → IDLE
                  ↓           ↓           ↓            ↓
               FAILED      FAILED      FAILED       FAILED

    State Transitions:
        - IDLE → SELFPLAY: Games needed, resources available
        - SELFPLAY → DATA_SYNC: Batch complete, threshold reached
        - DATA_SYNC → TRAINING: Data synced, ready to train
        - TRAINING → EVALUATION: Training complete, model ready
        - EVALUATION → PROMOTION: Model beats baselines
        - PROMOTION → IDLE: Model deployed, cycle complete
        - ANY → FAILED: Unrecoverable error

    Usage:
        from app.training.fault_tolerance import PipelineStage

        current_stage = PipelineStage.SELFPLAY

        # Check valid transitions
        if current_stage.can_transition_to(PipelineStage.DATA_SYNC):
            current_stage = PipelineStage.DATA_SYNC
    """
    IDLE = "idle"
    SELFPLAY = "selfplay"
    DATA_SYNC = "data_sync"
    TRAINING = "training"
    EVALUATION = "evaluation"
    PROMOTION = "promotion"
    FAILED = "failed"

    def can_transition_to(self, target: "PipelineStage") -> bool:
        """Check if transition to target stage is valid."""
        valid_transitions = {
            PipelineStage.IDLE: {PipelineStage.SELFPLAY, PipelineStage.FAILED},
            PipelineStage.SELFPLAY: {PipelineStage.DATA_SYNC, PipelineStage.FAILED},
            PipelineStage.DATA_SYNC: {PipelineStage.TRAINING, PipelineStage.FAILED},
            PipelineStage.TRAINING: {PipelineStage.EVALUATION, PipelineStage.FAILED},
            PipelineStage.EVALUATION: {PipelineStage.PROMOTION, PipelineStage.IDLE, PipelineStage.FAILED},
            PipelineStage.PROMOTION: {PipelineStage.IDLE, PipelineStage.FAILED},
            PipelineStage.FAILED: {PipelineStage.IDLE},  # Can restart from failed
        }
        return target in valid_transitions.get(self, set())

    def next_stage(self) -> "PipelineStage":
        """Get the next stage in normal progression."""
        progression = {
            PipelineStage.IDLE: PipelineStage.SELFPLAY,
            PipelineStage.SELFPLAY: PipelineStage.DATA_SYNC,
            PipelineStage.DATA_SYNC: PipelineStage.TRAINING,
            PipelineStage.TRAINING: PipelineStage.EVALUATION,
            PipelineStage.EVALUATION: PipelineStage.PROMOTION,
            PipelineStage.PROMOTION: PipelineStage.IDLE,
        }
        return progression.get(self, PipelineStage.IDLE)


# Only define CheckpointMetadata and TrainingProgress if not imported from checkpoint_unified
if _HAS_UNIFIED_CHECKPOINT:
    from app.training.checkpoint_unified import CheckpointMetadata, TrainingProgress
else:
    @dataclass
    class CheckpointMetadata:
        """Metadata for a checkpoint."""
        checkpoint_id: str
        checkpoint_type: CheckpointType
        epoch: int
        global_step: int
        timestamp: datetime
        metrics: dict[str, float]
        training_config: dict[str, Any]
        file_path: str
        file_hash: str
        parent_checkpoint: str | None = None

        def to_dict(self) -> dict[str, Any]:
            d = asdict(self)
            d['checkpoint_type'] = self.checkpoint_type.value
            d['timestamp'] = self.timestamp.isoformat()
            return d

        @classmethod
        def from_dict(cls, d: dict[str, Any]) -> 'CheckpointMetadata':
            d = d.copy()
            d['checkpoint_type'] = CheckpointType(d['checkpoint_type'])
            d['timestamp'] = datetime.fromisoformat(d['timestamp'])
            return cls(**d)


    @dataclass
    class TrainingProgress:
        """Tracks training progress for recovery."""
        epoch: int = 0
        global_step: int = 0
        batch_idx: int = 0
        samples_seen: int = 0
        best_metric: float | None = None
        best_metric_name: str = "loss"
        best_epoch: int = 0
        total_epochs: int = 100
        learning_rate: float = 0.001
        optimizer_state: dict[str, Any] | None = None
        scheduler_state: dict[str, Any] | None = None
        random_state: dict[str, Any] | None = None
        extra_state: dict[str, Any] = field(default_factory=dict)

        def to_dict(self) -> dict[str, Any]:
            return asdict(self)

        @classmethod
        def from_dict(cls, d: dict[str, Any]) -> 'TrainingProgress':
            return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class _LegacyCheckpointManager:
    """
    Legacy checkpoint manager (fallback when checkpoint_unified not available).

    Note: This is kept for backwards compatibility. New code should use
    UnifiedCheckpointManager from checkpoint_unified.py which provides
    additional features like adaptive checkpointing and architecture versioning.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 5,
        keep_best: int = 3,
        keep_every_n_epochs: int = 10
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        self.keep_every_n_epochs = keep_every_n_epochs

        self.metadata_file = self.checkpoint_dir / "checkpoints.json"
        self.checkpoints: list[CheckpointMetadata] = []

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
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load checkpoint metadata: {e}")
                self.checkpoints = []

    def _save_metadata(self):
        """Save checkpoint metadata to disk."""
        data = {
            'checkpoints': [c.to_dict() for c in self.checkpoints],
            'updated_at': datetime.now().isoformat()
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of checkpoint file."""
        return compute_file_checksum(file_path, truncate=16)

    def _generate_checkpoint_id(self, epoch: int, step: int, checkpoint_type: CheckpointType) -> str:
        """Generate unique checkpoint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ckpt_e{epoch:04d}_s{step:08d}_{checkpoint_type.value}_{timestamp}"

    def save_checkpoint(
        self,
        model_state: dict[str, Any],
        progress: TrainingProgress,
        checkpoint_type: CheckpointType = CheckpointType.REGULAR,
        metrics: dict[str, float] | None = None,
        training_config: dict[str, Any] | None = None
    ) -> CheckpointMetadata:
        """
        Save a training checkpoint.

        Args:
            model_state: Model state dict to save
            progress: Training progress state
            checkpoint_type: Type of checkpoint
            metrics: Current training metrics
            training_config: Training configuration

        Returns:
            CheckpointMetadata for the saved checkpoint
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for checkpoint saving")

        checkpoint_id = self._generate_checkpoint_id(
            progress.epoch, progress.global_step, checkpoint_type
        )
        file_path = self.checkpoint_dir / f"{checkpoint_id}.pt"

        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': model_state,
            'progress': progress.to_dict(),
            'checkpoint_type': checkpoint_type.value,
            'metrics': metrics or {},
            'training_config': training_config or {},
            'timestamp': datetime.now().isoformat()
        }

        # Save checkpoint
        torch.save(checkpoint_data, file_path)

        # Compute hash
        file_hash = self._compute_hash(file_path)

        # Create metadata
        parent = self.checkpoints[-1].checkpoint_id if self.checkpoints else None
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            epoch=progress.epoch,
            global_step=progress.global_step,
            timestamp=datetime.now(),
            metrics=metrics or {},
            training_config=training_config or {},
            file_path=str(file_path),
            file_hash=file_hash,
            parent_checkpoint=parent
        )

        self.checkpoints.append(metadata)
        self._save_metadata()

        # Cleanup old checkpoints
        self._cleanup_checkpoints(metrics)

        logger.info(f"Saved checkpoint: {checkpoint_id}")
        return metadata

    def _cleanup_checkpoints(self, current_metrics: dict[str, float] | None = None):
        """Remove old checkpoints based on retention policy."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return

        # Identify checkpoints to keep
        keep_ids = set()

        # Keep best checkpoints
        if current_metrics:
            sorted_by_loss = sorted(
                self.checkpoints,
                key=lambda c: c.metrics.get('loss', float('inf'))
            )
            for c in sorted_by_loss[:self.keep_best]:
                keep_ids.add(c.checkpoint_id)

        # Keep epoch checkpoints
        for c in self.checkpoints:
            if c.epoch % self.keep_every_n_epochs == 0:
                keep_ids.add(c.checkpoint_id)

        # Keep most recent
        for c in self.checkpoints[-self.max_checkpoints:]:
            keep_ids.add(c.checkpoint_id)

        # Keep emergency and best type checkpoints
        for c in self.checkpoints:
            if c.checkpoint_type in (CheckpointType.EMERGENCY, CheckpointType.BEST):
                keep_ids.add(c.checkpoint_id)

        # Remove checkpoints not in keep set
        to_remove = [c for c in self.checkpoints if c.checkpoint_id not in keep_ids]
        for c in to_remove:
            try:
                Path(c.file_path).unlink()
                logger.debug(f"Removed checkpoint: {c.checkpoint_id}")
            except OSError:
                pass
            self.checkpoints.remove(c)

        self._save_metadata()

    def load_checkpoint(
        self,
        checkpoint_id: str | None = None,
        checkpoint_type: CheckpointType | None = None
    ) -> dict[str, Any] | None:
        """
        Load a checkpoint.

        Args:
            checkpoint_id: Specific checkpoint to load
            checkpoint_type: Load latest checkpoint of this type

        Returns:
            Checkpoint data dict or None
        """
        try:
            import torch  # noqa: F401 - availability check
        except ImportError:
            raise ImportError("PyTorch required for checkpoint loading")

        metadata = None

        if checkpoint_id:
            for c in self.checkpoints:
                if c.checkpoint_id == checkpoint_id:
                    metadata = c
                    break
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

        # Verify hash
        computed_hash = self._compute_hash(file_path)
        if computed_hash != metadata.file_hash:
            logger.warning(f"Checkpoint hash mismatch for {metadata.checkpoint_id}")

        checkpoint_data = safe_load_checkpoint(file_path, map_location='cpu', warn_on_unsafe=False)
        checkpoint_data['metadata'] = metadata

        return checkpoint_data

    def get_best_checkpoint(self, metric_name: str = 'loss',
                            lower_is_better: bool = True) -> CheckpointMetadata | None:
        """Get the best checkpoint by a metric."""
        valid_checkpoints = [c for c in self.checkpoints if metric_name in c.metrics]
        if not valid_checkpoints:
            return None

        return min(valid_checkpoints, key=lambda c: c.metrics[metric_name] * (1 if lower_is_better else -1))

    def get_latest_checkpoint(self) -> CheckpointMetadata | None:
        """Get the most recent checkpoint."""
        return self.checkpoints[-1] if self.checkpoints else None

    def list_checkpoints(self) -> list[CheckpointMetadata]:
        """List all available checkpoints."""
        return list(self.checkpoints)


# Use UnifiedCheckpointManager when available (preferred), fallback to legacy
if _HAS_UNIFIED_CHECKPOINT and UnifiedCheckpointManager is not None:
    CheckpointManager = UnifiedCheckpointManager
else:
    CheckpointManager = _LegacyCheckpointManager  # type: ignore


class HeartbeatMonitor:
    """
    Monitors training process health via heartbeats.
    """

    def __init__(
        self,
        heartbeat_interval: float = 30.0,
        timeout_threshold: float = 120.0,
        on_timeout: Callable[[], None] | None = None
    ):
        self.heartbeat_interval = heartbeat_interval
        self.timeout_threshold = timeout_threshold
        self.on_timeout = on_timeout

        self.last_heartbeat: datetime | None = None
        self.heartbeat_file: Path | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

    def start(self, heartbeat_file: Path | None = None):
        """Start the heartbeat monitor."""
        self.heartbeat_file = heartbeat_file
        self._running = True
        self.beat()

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the heartbeat monitor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def beat(self):
        """Record a heartbeat."""
        with self._lock:
            self.last_heartbeat = datetime.now()

            if self.heartbeat_file:
                try:
                    data = {
                        'timestamp': self.last_heartbeat.isoformat(),
                        'pid': os.getpid()
                    }
                    with open(self.heartbeat_file, 'w') as f:
                        json.dump(data, f)
                except OSError:
                    pass

    def _monitor_loop(self):
        """Monitor heartbeats and detect timeouts."""
        while self._running:
            time.sleep(self.heartbeat_interval)

            with self._lock:
                if self.last_heartbeat:
                    elapsed = (datetime.now() - self.last_heartbeat).total_seconds()
                    if elapsed > self.timeout_threshold:
                        logger.warning(f"Heartbeat timeout: {elapsed:.1f}s since last heartbeat")
                        if self.on_timeout:
                            try:
                                self.on_timeout()
                            except Exception as e:
                                logger.error(f"Timeout handler error: {e}")

    @staticmethod
    def check_external_heartbeat(heartbeat_file: Path, timeout: float = 120.0) -> bool:
        """Check if an external process is still alive via heartbeat file."""
        if not heartbeat_file.exists():
            return False

        try:
            with open(heartbeat_file) as f:
                data = json.load(f)
            timestamp = datetime.fromisoformat(data['timestamp'])
            elapsed = (datetime.now() - timestamp).total_seconds()
            return elapsed < timeout
        except (json.JSONDecodeError, KeyError, OSError):
            return False


class GracefulShutdown:
    """
    Handles graceful shutdown signals for training processes.
    """

    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        self._shutdown_requested = False
        self._original_handlers: dict[int, Any] = {}
        self._model_state_getter: Callable[[], dict[str, Any]] | None = None
        self._progress_getter: Callable[[], TrainingProgress] | None = None

    def setup(
        self,
        model_state_getter: Callable[[], dict[str, Any]],
        progress_getter: Callable[[], TrainingProgress]
    ):
        """
        Setup signal handlers.

        Args:
            model_state_getter: Function to get current model state
            progress_getter: Function to get current training progress
        """
        self._model_state_getter = model_state_getter
        self._progress_getter = progress_getter

        # Install signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._original_handlers[sig] = signal.signal(sig, self._handle_signal)

    def teardown(self):
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        self._original_handlers.clear()

    def _handle_signal(self, signum: int, frame):
        """Handle shutdown signal."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True

        # Save emergency checkpoint
        if self._model_state_getter and self._progress_getter:
            try:
                self.checkpoint_manager.save_checkpoint(
                    model_state=self._model_state_getter(),
                    progress=self._progress_getter(),
                    checkpoint_type=CheckpointType.EMERGENCY,
                    metrics={'shutdown_signal': signum}
                )
                logger.info("Emergency checkpoint saved")
            except Exception as e:
                logger.error(f"Failed to save emergency checkpoint: {e}")

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown was requested."""
        return self._shutdown_requested

    def request_shutdown(self):
        """Programmatically request shutdown."""
        self._shutdown_requested = True


class FaultTolerantTrainer:
    """
    Wrapper for fault-tolerant training with automatic recovery.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_interval_steps: int = 1000,
        checkpoint_interval_epochs: int = 1,
        max_retries: int = 3,
        retry_delay: float = 10.0,
        heartbeat_interval: float = 30.0
    ):
        # Create checkpoint manager with config
        if _HAS_UNIFIED_CHECKPOINT and UnifiedCheckpointConfig is not None:
            config = UnifiedCheckpointConfig(checkpoint_dir=str(checkpoint_dir))
            self.checkpoint_manager = CheckpointManager(config)
        else:
            # Fallback to legacy checkpoint manager
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.checkpoint_interval_steps = checkpoint_interval_steps
        self.checkpoint_interval_epochs = checkpoint_interval_epochs
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.heartbeat_monitor = HeartbeatMonitor(
            heartbeat_interval=heartbeat_interval,
            on_timeout=self._on_heartbeat_timeout
        )

        self.graceful_shutdown = GracefulShutdown(self.checkpoint_manager)

        self.progress = TrainingProgress()
        self.state = TrainingState.INITIALIZING
        self._model_state: dict[str, Any] | None = None
        self._training_config: dict[str, Any] | None = None
        self._current_metrics: dict[str, float] = {}
        self._retry_count = 0

    def _on_heartbeat_timeout(self):
        """Handle heartbeat timeout."""
        logger.warning("Heartbeat timeout detected")
        self._save_emergency_checkpoint()

    def _save_emergency_checkpoint(self):
        """Save an emergency checkpoint."""
        if self._model_state:
            try:
                self.checkpoint_manager.save_checkpoint(
                    model_state=self._model_state,
                    progress=self.progress,
                    checkpoint_type=CheckpointType.EMERGENCY,
                    metrics=self._current_metrics,
                    training_config=self._training_config
                )
            except Exception as e:
                logger.error(f"Failed to save emergency checkpoint: {e}")

    def initialize(
        self,
        model_state: dict[str, Any],
        training_config: dict[str, Any],
        total_epochs: int,
        resume: bool = True
    ) -> TrainingProgress:
        """
        Initialize training, potentially resuming from checkpoint.

        Returns: TrainingProgress (either fresh or restored)
        """
        self._training_config = training_config
        self.progress.total_epochs = total_epochs

        if resume:
            checkpoint = self.checkpoint_manager.load_checkpoint()
            if checkpoint:
                logger.info(f"Resuming from checkpoint: {checkpoint['metadata'].checkpoint_id}")
                self._model_state = checkpoint['model_state_dict']
                self.progress = TrainingProgress.from_dict(checkpoint['progress'])
                self.state = TrainingState.RECOVERING
                return self.progress

        self._model_state = model_state
        self.state = TrainingState.RUNNING
        return self.progress

    def setup_signal_handling(
        self,
        model_state_getter: Callable[[], dict[str, Any]],
        progress_getter: Callable[[], TrainingProgress]
    ):
        """Setup graceful shutdown handling."""
        self.graceful_shutdown.setup(model_state_getter, progress_getter)

    def start_heartbeat(self, heartbeat_file: Path | None = None):
        """Start heartbeat monitoring."""
        self.heartbeat_monitor.start(heartbeat_file)

    def stop(self):
        """Stop all monitoring."""
        self.heartbeat_monitor.stop()
        self.graceful_shutdown.teardown()

    def should_checkpoint(self) -> bool:
        """Check if a checkpoint should be saved."""
        # Check step interval
        if self.progress.global_step > 0 and \
           self.progress.global_step % self.checkpoint_interval_steps == 0:
            return True

        # Check if epoch just completed
        return bool(self.progress.batch_idx == 0 and self.progress.epoch > 0 and self.progress.epoch % self.checkpoint_interval_epochs == 0)

    def update_progress(
        self,
        epoch: int,
        batch_idx: int,
        global_step: int,
        metrics: dict[str, float],
        model_state: dict[str, Any] | None = None
    ):
        """Update training progress."""
        self.progress.epoch = epoch
        self.progress.batch_idx = batch_idx
        self.progress.global_step = global_step
        self._current_metrics = metrics

        if model_state:
            self._model_state = model_state

        # Beat heartbeat
        self.heartbeat_monitor.beat()

        # Track best metric
        if self.progress.best_metric_name in metrics:
            current = metrics[self.progress.best_metric_name]
            if self.progress.best_metric is None or current < self.progress.best_metric:
                self.progress.best_metric = current
                self.progress.best_epoch = epoch

    def checkpoint_if_needed(
        self,
        model_state: dict[str, Any],
        force: bool = False,
        checkpoint_type: CheckpointType = CheckpointType.REGULAR
    ) -> CheckpointMetadata | None:
        """Save checkpoint if conditions are met or forced."""
        if not force and not self.should_checkpoint():
            return None

        self._model_state = model_state
        self.state = TrainingState.CHECKPOINTING

        try:
            metadata = self.checkpoint_manager.save_checkpoint(
                model_state=model_state,
                progress=self.progress,
                checkpoint_type=checkpoint_type,
                metrics=self._current_metrics,
                training_config=self._training_config
            )
            self.state = TrainingState.RUNNING
            return metadata
        except Exception as e:
            logger.error(f"Checkpoint failed: {e}")
            self.state = TrainingState.RUNNING
            return None

    def save_best_checkpoint(self, model_state: dict[str, Any],
                             metric_name: str, metric_value: float) -> CheckpointMetadata | None:
        """Save a checkpoint if this is the best metric so far."""
        if self.progress.best_metric is None or metric_value < self.progress.best_metric:
            self.progress.best_metric = metric_value
            self.progress.best_metric_name = metric_name
            self.progress.best_epoch = self.progress.epoch

            return self.checkpoint_manager.save_checkpoint(
                model_state=model_state,
                progress=self.progress,
                checkpoint_type=CheckpointType.BEST,
                metrics={metric_name: metric_value, **self._current_metrics},
                training_config=self._training_config
            )
        return None

    @property
    def should_stop(self) -> bool:
        """Check if training should stop."""
        return self.graceful_shutdown.shutdown_requested

    @contextmanager
    def fault_tolerant_epoch(self, epoch: int):
        """Context manager for fault-tolerant epoch execution."""
        self.progress.epoch = epoch
        self.progress.batch_idx = 0

        try:
            yield
            self._retry_count = 0  # Reset on successful epoch
        except Exception as e:
            self._retry_count += 1
            logger.error(f"Epoch {epoch} failed (attempt {self._retry_count}): {e}")
            traceback.print_exc()

            if self._retry_count < self.max_retries:
                logger.info(f"Retrying in {self.retry_delay}s...")
                time.sleep(self.retry_delay)
                # Will retry from last checkpoint
            else:
                logger.error("Max retries exceeded, saving emergency checkpoint")
                self._save_emergency_checkpoint()
                self.state = TrainingState.FAILED
                raise

    def complete(self, model_state: dict[str, Any]) -> CheckpointMetadata:
        """Mark training as complete and save final checkpoint."""
        self.state = TrainingState.COMPLETED
        return self.checkpoint_manager.save_checkpoint(
            model_state=model_state,
            progress=self.progress,
            checkpoint_type=CheckpointType.EPOCH,
            metrics=self._current_metrics,
            training_config=self._training_config
        )


class DistributedFaultHandler:
    """
    Handles faults in distributed training scenarios.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        coordinator_path: Path,
        timeout: float = 300.0
    ):
        self.world_size = world_size
        self.rank = rank
        self.coordinator_path = Path(coordinator_path)
        self.coordinator_path.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

        self._heartbeat_file = self.coordinator_path / f"worker_{rank}_heartbeat.json"
        self._status_file = self.coordinator_path / f"worker_{rank}_status.json"
        self._barrier_dir = self.coordinator_path / "barriers"
        self._barrier_dir.mkdir(exist_ok=True)

    def report_status(self, status: str, epoch: int, step: int):
        """Report worker status."""
        data = {
            'rank': self.rank,
            'status': status,
            'epoch': epoch,
            'step': step,
            'timestamp': datetime.now().isoformat()
        }
        with open(self._status_file, 'w') as f:
            json.dump(data, f)

    def report_heartbeat(self):
        """Report heartbeat."""
        data = {
            'rank': self.rank,
            'timestamp': datetime.now().isoformat(),
            'pid': os.getpid()
        }
        with open(self._heartbeat_file, 'w') as f:
            json.dump(data, f)

    def check_all_workers_alive(self, timeout: float | None = None) -> list[int]:
        """Check which workers are alive. Returns list of dead worker ranks."""
        timeout = timeout or self.timeout
        dead_workers = []

        for rank in range(self.world_size):
            heartbeat_file = self.coordinator_path / f"worker_{rank}_heartbeat.json"
            if not HeartbeatMonitor.check_external_heartbeat(heartbeat_file, timeout):
                dead_workers.append(rank)

        return dead_workers

    def barrier(self, name: str, timeout: float | None = None) -> bool:
        """
        File-based barrier synchronization.

        Returns True if all workers reached barrier, False on timeout.
        """
        timeout = timeout or self.timeout
        barrier_file = self._barrier_dir / f"{name}_{self.rank}.barrier"

        # Signal arrival at barrier
        with open(barrier_file, 'w') as f:
            f.write(datetime.now().isoformat())

        # Wait for all workers
        start_time = time.time()
        while time.time() - start_time < timeout:
            arrived = 0
            for rank in range(self.world_size):
                rank_barrier = self._barrier_dir / f"{name}_{rank}.barrier"
                if rank_barrier.exists():
                    arrived += 1

            if arrived == self.world_size:
                # Cleanup barrier files
                if self.rank == 0:
                    time.sleep(0.5)  # Give other workers time to see completion
                    for rank in range(self.world_size):
                        with suppress(OSError):
                            (self._barrier_dir / f"{name}_{rank}.barrier").unlink()
                return True

            time.sleep(0.1)

        logger.warning(f"Barrier {name} timed out")
        return False

    def elect_leader(self) -> int:
        """Elect a new leader among alive workers. Returns leader rank."""
        dead_workers = self.check_all_workers_alive()
        alive_workers = [r for r in range(self.world_size) if r not in dead_workers]

        if not alive_workers:
            raise RuntimeError("No alive workers!")

        return min(alive_workers)


def main():
    """Example usage of fault tolerance features."""
    import tempfile

    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

        def forward(self, x):
            return self.fc(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        # Create fault-tolerant trainer
        trainer = FaultTolerantTrainer(
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval_steps=100,
            checkpoint_interval_epochs=1
        )

        # Create model
        model = DummyModel()
        training_config = {'learning_rate': 0.001, 'batch_size': 32}

        # Initialize training (potentially resuming)
        progress = trainer.initialize(
            model_state=model.state_dict(),
            training_config=training_config,
            total_epochs=10,
            resume=True
        )

        print(f"Starting from epoch {progress.epoch}")

        # Setup signal handling
        trainer.setup_signal_handling(
            model_state_getter=lambda: model.state_dict(),
            progress_getter=lambda: trainer.progress
        )

        # Start heartbeat monitoring
        trainer.start_heartbeat()

        try:
            for epoch in range(progress.epoch, progress.total_epochs):
                with trainer.fault_tolerant_epoch(epoch):
                    print(f"Epoch {epoch}")

                    # Simulate training
                    for batch_idx in range(100):
                        if trainer.should_stop:
                            print("Shutdown requested")
                            break

                        # Update progress
                        trainer.update_progress(
                            epoch=epoch,
                            batch_idx=batch_idx,
                            global_step=epoch * 100 + batch_idx,
                            metrics={'loss': 1.0 / (epoch + 1), 'accuracy': 0.5 + epoch * 0.05}
                        )

                        # Checkpoint if needed
                        trainer.checkpoint_if_needed(model.state_dict())

                    if trainer.should_stop:
                        break

                    # End of epoch checkpoint
                    trainer.checkpoint_if_needed(
                        model.state_dict(),
                        force=True,
                        checkpoint_type=CheckpointType.EPOCH
                    )

                    # Save best checkpoint
                    trainer.save_best_checkpoint(
                        model.state_dict(),
                        'loss',
                        1.0 / (epoch + 1)
                    )

            # Complete training
            if not trainer.should_stop:
                trainer.complete(model.state_dict())
                print("Training completed")

        finally:
            trainer.stop()

        # List checkpoints
        print("\nCheckpoints:")
        for ckpt in trainer.checkpoint_manager.list_checkpoints():
            print(f"  {ckpt.checkpoint_id} ({ckpt.checkpoint_type.value})")

        # Get best checkpoint
        best = trainer.checkpoint_manager.get_best_checkpoint('loss')
        if best:
            print(f"\nBest checkpoint: {best.checkpoint_id} (loss={best.metrics.get('loss', 'N/A')})")


# =============================================================================
# Consolidated Exports (December 2025)
# =============================================================================
# Re-export training-specific exception handling from exception_integration
# for convenient access through fault_tolerance module

try:
    from app.training.exception_integration import (
        # Training-specific retry policies
        TrainingRetryPolicies,
    )
    _HAS_EXCEPTION_INTEGRATION = True
except ImportError:
    _HAS_EXCEPTION_INTEGRATION = False
    TrainingRetryPolicies = None  # type: ignore


if __name__ == "__main__":
    main()
