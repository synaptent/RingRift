"""Exception Handling Integration for Training Components.

Provides specialized retry policies and exception handling for training operations:
- Training step retries
- Checkpoint save/load retries
- Evaluation retries
- Selfplay retries
- Data loading retries

Usage:
    from app.training.exception_integration import (
        TrainingRetryPolicies,
        retry_checkpoint_save,
        retry_data_load,
        safe_training_step,
    )

    # Use predefined retry policies
    @TrainingRetryPolicies.checkpoint_save()
    def save_checkpoint():
        ...

    # Or use specialized decorators
    @retry_checkpoint_save
    def my_checkpoint_save():
        ...

    # Safe execution for training steps
    result = safe_training_step(train_one_step, model, batch)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from app.core.error_handler import (
    ErrorAggregator,
    RetryPolicy,
    RetryStrategy,
    retry,
    safe_execute,
)
from app.errors import (
    FatalError,
)

logger = logging.getLogger(__name__)

# Type vars for decorators
F = TypeVar("F", bound=Callable[..., Any])
AF = TypeVar("AF", bound=Callable[..., Any])

__all__ = [
    "CheckpointError",
    "DataLoadError",
    "EvaluationError",
    "SelfplayError",
    # Exception types
    "TrainingError",
    # Error aggregation
    "TrainingErrorAggregator",
    # Retry policies
    "TrainingRetryPolicies",
    "retry_checkpoint_load",
    # Decorators
    "retry_checkpoint_save",
    "retry_data_load",
    "retry_evaluation",
    "retry_selfplay",
    "retry_training_step",
    "safe_checkpoint_save",
    "safe_evaluation",
    # Safe execution
    "safe_training_step",
]


# =============================================================================
# Training-Specific Exception Types
# =============================================================================
# NOTE (December 2025): Import from canonical source app.errors
# These re-exports maintain backward compatibility

from app.errors import (
    CheckpointError,
    DataLoadError,
    EvaluationError,
    SelfplayError,
    TrainingError,
)

# =============================================================================
# Training Retry Policies
# =============================================================================

class TrainingRetryPolicies:
    """Predefined retry policies for training operations."""

    # Checkpoint save - critical operation, moderate retries
    CHECKPOINT_SAVE = RetryPolicy(
        strategy=RetryStrategy.EXPONENTIAL_JITTER,
        max_attempts=3,
        initial_delay=2.0,
        multiplier=2.0,
        max_delay=30.0,
        jitter_factor=0.1,
    )

    # Checkpoint load - quick retries for I/O issues
    CHECKPOINT_LOAD = RetryPolicy(
        strategy=RetryStrategy.EXPONENTIAL,
        max_attempts=3,
        initial_delay=1.0,
        multiplier=2.0,
        max_delay=10.0,
    )

    # Data loading - quick retries with jitter
    DATA_LOAD = RetryPolicy(
        strategy=RetryStrategy.EXPONENTIAL_JITTER,
        max_attempts=3,
        initial_delay=0.5,
        multiplier=2.0,
        max_delay=10.0,
        jitter_factor=0.2,
    )

    # Evaluation - longer delays, games can take time
    EVALUATION = RetryPolicy(
        strategy=RetryStrategy.EXPONENTIAL,
        max_attempts=2,
        initial_delay=5.0,
        multiplier=2.0,
        max_delay=60.0,
    )

    # Selfplay - very long running, fewer retries
    SELFPLAY = RetryPolicy(
        strategy=RetryStrategy.EXPONENTIAL,
        max_attempts=2,
        initial_delay=10.0,
        multiplier=2.0,
        max_delay=120.0,
    )

    # Training step - quick retries for transient GPU errors
    TRAINING_STEP = RetryPolicy(
        strategy=RetryStrategy.LINEAR,
        max_attempts=2,
        initial_delay=0.5,
        multiplier=1.0,
        max_delay=2.0,
    )

    # Model promotion - critical operation
    PROMOTION = RetryPolicy(
        strategy=RetryStrategy.EXPONENTIAL_JITTER,
        max_attempts=3,
        initial_delay=2.0,
        multiplier=2.0,
        max_delay=30.0,
        jitter_factor=0.15,
    )

    # Remote sync - network operations
    REMOTE_SYNC = RetryPolicy(
        strategy=RetryStrategy.EXPONENTIAL_JITTER,
        max_attempts=5,
        initial_delay=1.0,
        multiplier=2.0,
        max_delay=60.0,
        jitter_factor=0.2,
    )

    @classmethod
    def checkpoint_save(
        cls,
        exceptions: Sequence[type[Exception]] = (IOError, OSError, CheckpointError),
    ) -> Callable[[F], F]:
        """Get checkpoint save retry decorator.

        Args:
            exceptions: Exceptions to retry on

        Returns:
            Retry decorator
        """
        return retry(
            **cls.CHECKPOINT_SAVE.to_retry_kwargs(),
            exceptions=exceptions,
        )

    @classmethod
    def checkpoint_load(
        cls,
        exceptions: Sequence[type[Exception]] = (IOError, OSError, CheckpointError),
    ) -> Callable[[F], F]:
        """Get checkpoint load retry decorator."""
        return retry(
            **cls.CHECKPOINT_LOAD.to_retry_kwargs(),
            exceptions=exceptions,
        )

    @classmethod
    def data_load(
        cls,
        exceptions: Sequence[type[Exception]] = (IOError, OSError, DataLoadError),
    ) -> Callable[[F], F]:
        """Get data load retry decorator."""
        return retry(
            **cls.DATA_LOAD.to_retry_kwargs(),
            exceptions=exceptions,
        )

    @classmethod
    def evaluation(
        cls,
        exceptions: Sequence[type[Exception]] = (EvaluationError, RuntimeError),
    ) -> Callable[[F], F]:
        """Get evaluation retry decorator."""
        return retry(
            **cls.EVALUATION.to_retry_kwargs(),
            exceptions=exceptions,
        )

    @classmethod
    def selfplay(
        cls,
        exceptions: Sequence[type[Exception]] = (SelfplayError, RuntimeError),
    ) -> Callable[[F], F]:
        """Get selfplay retry decorator."""
        return retry(
            **cls.SELFPLAY.to_retry_kwargs(),
            exceptions=exceptions,
        )

    @classmethod
    def training_step(
        cls,
        exceptions: Sequence[type[Exception]] = (RuntimeError,),
    ) -> Callable[[F], F]:
        """Get training step retry decorator."""
        return retry(
            **cls.TRAINING_STEP.to_retry_kwargs(),
            exceptions=exceptions,
        )


# =============================================================================
# Convenience Decorators
# =============================================================================

def retry_checkpoint_save(func: F) -> F:
    """Decorator for retrying checkpoint save operations.

    Example:
        @retry_checkpoint_save
        def save_checkpoint(path, model, optimizer):
            torch.save({...}, path)
    """
    return TrainingRetryPolicies.checkpoint_save()(func)


def retry_checkpoint_load(func: F) -> F:
    """Decorator for retrying checkpoint load operations.

    Example:
        from app.utils.torch_utils import safe_load_checkpoint

        @retry_checkpoint_load
        def load_checkpoint(path):
            return safe_load_checkpoint(path)  # Use safe_load_checkpoint, not torch.load
    """
    return TrainingRetryPolicies.checkpoint_load()(func)


def retry_data_load(func: F) -> F:
    """Decorator for retrying data load operations.

    Example:
        @retry_data_load
        def load_batch(dataloader):
            return next(iter(dataloader))
    """
    return TrainingRetryPolicies.data_load()(func)


def retry_evaluation(func: F) -> F:
    """Decorator for retrying evaluation operations.

    Example:
        @retry_evaluation
        def run_evaluation(model, test_data):
            return evaluate(model, test_data)
    """
    return TrainingRetryPolicies.evaluation()(func)


def retry_selfplay(func: F) -> F:
    """Decorator for retrying selfplay operations.

    Example:
        @retry_selfplay
        def run_selfplay(config, iteration):
            return generate_games(config, iteration)
    """
    return TrainingRetryPolicies.selfplay()(func)


def retry_training_step(func: F) -> F:
    """Decorator for retrying training step operations.

    Example:
        @retry_training_step
        def train_step(model, optimizer, batch):
            loss = model(batch)
            loss.backward()
            optimizer.step()
            return loss.item()
    """
    return TrainingRetryPolicies.training_step()(func)


# =============================================================================
# Safe Execution Wrappers
# =============================================================================

def safe_training_step(
    func: Callable[..., Any],
    *args,
    default: Any = None,
    log_errors: bool = True,
    **kwargs,
) -> Any:
    """Execute a training step safely.

    Catches CUDA errors and other training exceptions.

    Args:
        func: Training step function
        *args: Positional arguments
        default: Default value on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments

    Returns:
        Result or default on error
    """
    try:
        return func(*args, **kwargs)
    except RuntimeError as e:
        # Check for CUDA OOM
        if "out of memory" in str(e).lower():
            if log_errors:
                logger.warning(f"CUDA OOM in {func.__name__}: {e}")
            # Try to free memory
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        elif log_errors:
            logger.warning(f"Training step failed: {func.__name__}: {e}")
        return default
    except Exception as e:
        if log_errors:
            logger.warning(f"Training step failed: {func.__name__}: {e}")
        return default


def safe_checkpoint_save(
    func: Callable[..., Any],
    *args,
    log_errors: bool = True,
    **kwargs,
) -> bool:
    """Execute a checkpoint save safely.

    Args:
        func: Checkpoint save function
        *args: Positional arguments
        log_errors: Whether to log errors
        **kwargs: Keyword arguments

    Returns:
        True if successful, False otherwise
    """
    try:
        func(*args, **kwargs)
        return True
    except Exception as e:
        if log_errors:
            logger.error(f"Checkpoint save failed: {func.__name__}: {e}")
        return False


def safe_evaluation(
    func: Callable[..., Any],
    *args,
    default: Any = None,
    log_errors: bool = True,
    **kwargs,
) -> Any:
    """Execute an evaluation safely.

    Args:
        func: Evaluation function
        *args: Positional arguments
        default: Default value on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments

    Returns:
        Result or default on error
    """
    return safe_execute(func, *args, default=default, log_errors=log_errors, **kwargs)


# =============================================================================
# Training Error Aggregator
# =============================================================================

class TrainingErrorAggregator(ErrorAggregator):
    """Error aggregator specialized for training operations.

    Extends ErrorAggregator with training-specific functionality.

    Usage:
        errors = TrainingErrorAggregator("training epoch 5")

        for batch in dataloader:
            try:
                loss = train_step(batch)
            except Exception as e:
                errors.add(e, context={"batch_idx": batch_idx})
                if errors.should_abort():
                    break

        if errors.has_errors:
            logger.warning(errors.summary())
    """

    def __init__(
        self,
        operation: str,
        max_errors_before_abort: int = 10,
    ):
        """Initialize training error aggregator.

        Args:
            operation: Description of the operation
            max_errors_before_abort: Max errors before recommending abort
        """
        super().__init__(operation)
        self.max_errors_before_abort = max_errors_before_abort

    def should_abort(self) -> bool:
        """Check if training should abort due to too many errors."""
        return self.count >= self.max_errors_before_abort

    def has_fatal_error(self) -> bool:
        """Check if any fatal (non-retryable) errors occurred."""
        return any(isinstance(e, FatalError) for e, _ in self.errors)

    def has_cuda_oom(self) -> bool:
        """Check if any CUDA OOM errors occurred."""
        return any(isinstance(e, RuntimeError) and "out of memory" in str(e).lower() for e, _ in self.errors)

    def get_error_types(self) -> dict[str, int]:
        """Get count of errors by type."""
        counts: dict[str, int] = {}
        for e, _ in self.errors:
            error_type = type(e).__name__
            counts[error_type] = counts.get(error_type, 0) + 1
        return counts


# =============================================================================
# Context Managers for Error Handling
# =============================================================================

from contextlib import contextmanager


@contextmanager
def training_error_context(
    operation: str,
    reraise: bool = True,
    log_level: str = "error",
):
    """Context manager for training error handling.

    Args:
        operation: Description of the operation
        reraise: Whether to reraise exceptions
        log_level: Log level for errors ("error", "warning", "debug")

    Yields:
        None

    Example:
        with training_error_context("checkpoint save"):
            save_checkpoint(path, model)
    """
    try:
        yield
    except FatalError:
        # Always reraise fatal errors
        raise
    except Exception as e:
        log_func = getattr(logger, log_level, logger.error)
        log_func(f"{operation} failed: {e}")

        if reraise:
            raise TrainingError(f"{operation} failed: {e}") from e


@contextmanager
def checkpoint_error_context(path: str, reraise: bool = True):
    """Context manager for checkpoint operations.

    Args:
        path: Checkpoint path
        reraise: Whether to reraise exceptions

    Yields:
        None
    """
    try:
        yield
    except Exception as e:
        logger.error(f"Checkpoint operation failed for {path}: {e}")
        if reraise:
            raise CheckpointError(f"Checkpoint operation failed for {path}: {e}") from e


@contextmanager
def evaluation_error_context(config_key: str, reraise: bool = True):
    """Context manager for evaluation operations.

    Args:
        config_key: Configuration key
        reraise: Whether to reraise exceptions

    Yields:
        None
    """
    try:
        yield
    except Exception as e:
        logger.error(f"Evaluation failed for {config_key}: {e}")
        if reraise:
            raise EvaluationError(f"Evaluation failed for {config_key}: {e}") from e
