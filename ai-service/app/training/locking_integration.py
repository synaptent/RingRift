"""Locking Integration for Training Components.

Provides specialized distributed locks for training operations:
- Checkpoint save/load protection
- Model registry access
- Training job coordination
- Selfplay coordination
- Evaluation serialization

Usage:
    from app.training.locking_integration import (
        checkpoint_lock,
        model_registry_lock,
        training_job_lock,
        TrainingLocks,
    )

    # Checkpoint protection
    with checkpoint_lock("square8_2p") as lock:
        if lock:
            save_checkpoint()

    # Model registry protection
    with model_registry_lock("square8_2p"):
        update_model_registry()

    # Or use the TrainingLocks class
    with TrainingLocks.checkpoint("square8_2p", step=1000):
        save_checkpoint()
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager

from app.coordination.distributed_lock import (
    DEFAULT_ACQUIRE_TIMEOUT,
    DEFAULT_LOCK_TIMEOUT,
    DistributedLock,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Lock types
    "TrainingLockType",
    # Lock helpers
    "TrainingLocks",
    # Context managers
    "checkpoint_lock",
    "evaluation_lock",
    # Utilities
    "get_training_lock",
    "is_training_locked",
    "model_registry_lock",
    "promotion_lock",
    "selfplay_lock",
    "training_job_lock",
]


# =============================================================================
# Lock Type Constants
# =============================================================================

class TrainingLockType:
    """Lock type prefixes for training components."""

    CHECKPOINT = "checkpoint"
    MODEL_REGISTRY = "model_registry"
    TRAINING_JOB = "training_job"
    EVALUATION = "evaluation"
    SELFPLAY = "selfplay"
    PROMOTION = "promotion"
    DATA_SYNC = "data_sync"


# =============================================================================
# Training Locks Class
# =============================================================================

class TrainingLocks:
    """Static class for acquiring training-specific locks.

    Provides a clean interface for common training lock patterns.
    All methods are static for easy use without initialization.
    """

    # -------------------------------------------------------------------------
    # Checkpoint Locks
    # -------------------------------------------------------------------------

    @staticmethod
    @contextmanager
    def checkpoint(
        config_key: str,
        step: int | None = None,
        timeout: int = 60,
    ) -> Generator[DistributedLock | None, None, None]:
        """Acquire a checkpoint lock.

        Protects checkpoint save/load operations from concurrent access.

        Args:
            config_key: Configuration key
            step: Optional step number (for step-specific locks)
            timeout: Acquire timeout in seconds

        Yields:
            Lock instance if acquired, None otherwise

        Example:
            with TrainingLocks.checkpoint("square8_2p", step=1000) as lock:
                if lock:
                    save_checkpoint(step=1000)
                else:
                    logger.warning("Could not acquire checkpoint lock")
        """
        lock_name = f"{TrainingLockType.CHECKPOINT}:{config_key}"
        if step is not None:
            lock_name = f"{lock_name}:step_{step}"

        lock = DistributedLock(lock_name, lock_timeout=DEFAULT_LOCK_TIMEOUT)
        acquired = lock.acquire(timeout=timeout)

        try:
            yield lock if acquired else None
        finally:
            if acquired:
                lock.release()

    @staticmethod
    @contextmanager
    def checkpoint_save(
        config_key: str,
        timeout: int = 120,
    ) -> Generator[DistributedLock | None, None, None]:
        """Acquire a lock specifically for checkpoint saving.

        Uses a longer timeout as checkpoint saves can be slow.

        Args:
            config_key: Configuration key
            timeout: Acquire timeout in seconds

        Yields:
            Lock instance if acquired, None otherwise
        """
        lock_name = f"{TrainingLockType.CHECKPOINT}:{config_key}:save"
        lock = DistributedLock(lock_name, lock_timeout=3600)  # 1 hour max
        acquired = lock.acquire(timeout=timeout)

        try:
            yield lock if acquired else None
        finally:
            if acquired:
                lock.release()

    # -------------------------------------------------------------------------
    # Model Registry Locks
    # -------------------------------------------------------------------------

    @staticmethod
    @contextmanager
    def model_registry(
        config_key: str,
        timeout: int = 30,
    ) -> Generator[DistributedLock | None, None, None]:
        """Acquire a model registry lock.

        Protects model registration and metadata updates.

        Args:
            config_key: Configuration key
            timeout: Acquire timeout in seconds

        Yields:
            Lock instance if acquired, None otherwise
        """
        lock_name = f"{TrainingLockType.MODEL_REGISTRY}:{config_key}"
        lock = DistributedLock(lock_name, lock_timeout=300)  # 5 min max
        acquired = lock.acquire(timeout=timeout)

        try:
            yield lock if acquired else None
        finally:
            if acquired:
                lock.release()

    # -------------------------------------------------------------------------
    # Training Job Locks
    # -------------------------------------------------------------------------

    @staticmethod
    @contextmanager
    def training_job(
        config_key: str,
        job_id: str | None = None,
        timeout: int = DEFAULT_ACQUIRE_TIMEOUT,
    ) -> Generator[DistributedLock | None, None, None]:
        """Acquire a training job lock.

        Ensures only one training job runs for a config at a time.

        Args:
            config_key: Configuration key
            job_id: Optional job ID for finer-grained locking
            timeout: Acquire timeout in seconds

        Yields:
            Lock instance if acquired, None otherwise
        """
        lock_name = f"{TrainingLockType.TRAINING_JOB}:{config_key}"
        if job_id:
            lock_name = f"{lock_name}:{job_id}"

        lock = DistributedLock(lock_name, lock_timeout=DEFAULT_LOCK_TIMEOUT)
        acquired = lock.acquire(timeout=timeout)

        try:
            yield lock if acquired else None
        finally:
            if acquired:
                lock.release()

    # -------------------------------------------------------------------------
    # Evaluation Locks
    # -------------------------------------------------------------------------

    @staticmethod
    @contextmanager
    def evaluation(
        config_key: str,
        timeout: int = 30,
    ) -> Generator[DistributedLock | None, None, None]:
        """Acquire an evaluation lock.

        Serializes evaluation runs to prevent resource contention.

        Args:
            config_key: Configuration key
            timeout: Acquire timeout in seconds

        Yields:
            Lock instance if acquired, None otherwise
        """
        lock_name = f"{TrainingLockType.EVALUATION}:{config_key}"
        lock = DistributedLock(lock_name, lock_timeout=1800)  # 30 min max
        acquired = lock.acquire(timeout=timeout)

        try:
            yield lock if acquired else None
        finally:
            if acquired:
                lock.release()

    # -------------------------------------------------------------------------
    # Selfplay Locks
    # -------------------------------------------------------------------------

    @staticmethod
    @contextmanager
    def selfplay(
        config_key: str,
        iteration: int | None = None,
        timeout: int = 30,
    ) -> Generator[DistributedLock | None, None, None]:
        """Acquire a selfplay lock.

        Coordinates selfplay generation across nodes.

        Args:
            config_key: Configuration key
            iteration: Optional iteration for finer-grained locking
            timeout: Acquire timeout in seconds

        Yields:
            Lock instance if acquired, None otherwise
        """
        lock_name = f"{TrainingLockType.SELFPLAY}:{config_key}"
        if iteration is not None:
            lock_name = f"{lock_name}:iter_{iteration}"

        lock = DistributedLock(lock_name, lock_timeout=7200)  # 2 hour max
        acquired = lock.acquire(timeout=timeout)

        try:
            yield lock if acquired else None
        finally:
            if acquired:
                lock.release()

    # -------------------------------------------------------------------------
    # Promotion Locks
    # -------------------------------------------------------------------------

    @staticmethod
    @contextmanager
    def promotion(
        config_key: str,
        timeout: int = 60,
    ) -> Generator[DistributedLock | None, None, None]:
        """Acquire a model promotion lock.

        Ensures atomic model promotions.

        Args:
            config_key: Configuration key
            timeout: Acquire timeout in seconds

        Yields:
            Lock instance if acquired, None otherwise
        """
        lock_name = f"{TrainingLockType.PROMOTION}:{config_key}"
        lock = DistributedLock(lock_name, lock_timeout=600)  # 10 min max
        acquired = lock.acquire(timeout=timeout)

        try:
            yield lock if acquired else None
        finally:
            if acquired:
                lock.release()

    # -------------------------------------------------------------------------
    # Data Sync Locks
    # -------------------------------------------------------------------------

    @staticmethod
    @contextmanager
    def data_sync(
        config_key: str,
        timeout: int = 30,
    ) -> Generator[DistributedLock | None, None, None]:
        """Acquire a data sync lock.

        Protects data synchronization operations.

        Args:
            config_key: Configuration key
            timeout: Acquire timeout in seconds

        Yields:
            Lock instance if acquired, None otherwise
        """
        lock_name = f"{TrainingLockType.DATA_SYNC}:{config_key}"
        lock = DistributedLock(lock_name, lock_timeout=300)  # 5 min max
        acquired = lock.acquire(timeout=timeout)

        try:
            yield lock if acquired else None
        finally:
            if acquired:
                lock.release()


# =============================================================================
# Convenience Context Managers
# =============================================================================

@contextmanager
def checkpoint_lock(
    config_key: str,
    step: int | None = None,
    timeout: int = 60,
) -> Generator[DistributedLock | None, None, None]:
    """Context manager for checkpoint locking.

    Example:
        with checkpoint_lock("square8_2p", step=1000) as lock:
            if lock:
                save_checkpoint()
    """
    with TrainingLocks.checkpoint(config_key, step, timeout) as lock:
        yield lock


@contextmanager
def model_registry_lock(
    config_key: str,
    timeout: int = 30,
) -> Generator[DistributedLock | None, None, None]:
    """Context manager for model registry locking.

    Example:
        with model_registry_lock("square8_2p") as lock:
            if lock:
                register_model()
    """
    with TrainingLocks.model_registry(config_key, timeout) as lock:
        yield lock


@contextmanager
def training_job_lock(
    config_key: str,
    timeout: int = DEFAULT_ACQUIRE_TIMEOUT,
) -> Generator[DistributedLock | None, None, None]:
    """Context manager for training job locking.

    Example:
        with training_job_lock("square8_2p") as lock:
            if lock:
                run_training()
    """
    with TrainingLocks.training_job(config_key, timeout=timeout) as lock:
        yield lock


@contextmanager
def evaluation_lock(
    config_key: str,
    timeout: int = 30,
) -> Generator[DistributedLock | None, None, None]:
    """Context manager for evaluation locking.

    Example:
        with evaluation_lock("square8_2p") as lock:
            if lock:
                run_evaluation()
    """
    with TrainingLocks.evaluation(config_key, timeout) as lock:
        yield lock


@contextmanager
def selfplay_lock(
    config_key: str,
    iteration: int | None = None,
    timeout: int = 30,
) -> Generator[DistributedLock | None, None, None]:
    """Context manager for selfplay locking.

    Example:
        with selfplay_lock("square8_2p", iteration=10) as lock:
            if lock:
                run_selfplay()
    """
    with TrainingLocks.selfplay(config_key, iteration, timeout) as lock:
        yield lock


@contextmanager
def promotion_lock(
    config_key: str,
    timeout: int = 60,
) -> Generator[DistributedLock | None, None, None]:
    """Context manager for promotion locking.

    Example:
        with promotion_lock("square8_2p") as lock:
            if lock:
                promote_model()
    """
    with TrainingLocks.promotion(config_key, timeout) as lock:
        yield lock


# =============================================================================
# Utilities
# =============================================================================

def get_training_lock(
    lock_type: str,
    config_key: str,
    **kwargs,
) -> DistributedLock:
    """Get a distributed lock for a training component.

    Args:
        lock_type: Lock type from TrainingLockType
        config_key: Configuration key
        **kwargs: Additional arguments for lock

    Returns:
        DistributedLock instance (not acquired)
    """
    lock_name = f"{lock_type}:{config_key}"
    return DistributedLock(lock_name, **kwargs)


def is_training_locked(
    lock_type: str,
    config_key: str,
) -> bool:
    """Check if a training lock is currently held.

    Args:
        lock_type: Lock type from TrainingLockType
        config_key: Configuration key

    Returns:
        True if lock is held
    """
    lock = get_training_lock(lock_type, config_key)
    return lock.is_locked()


def get_lock_status(config_key: str) -> dict:
    """Get status of all locks for a config.

    Args:
        config_key: Configuration key

    Returns:
        Dict with lock type -> is_locked status
    """
    return {
        "checkpoint": is_training_locked(TrainingLockType.CHECKPOINT, config_key),
        "model_registry": is_training_locked(TrainingLockType.MODEL_REGISTRY, config_key),
        "training_job": is_training_locked(TrainingLockType.TRAINING_JOB, config_key),
        "evaluation": is_training_locked(TrainingLockType.EVALUATION, config_key),
        "selfplay": is_training_locked(TrainingLockType.SELFPLAY, config_key),
        "promotion": is_training_locked(TrainingLockType.PROMOTION, config_key),
        "data_sync": is_training_locked(TrainingLockType.DATA_SYNC, config_key),
    }
