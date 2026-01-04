"""Unified Resilience Orchestrator (Phase 2.1 - January 2026).

Provides centralized retry/backoff/circuit-breaker coordination for all
coordination layer operations. This consolidates scattered resilience patterns
across 176+ files into a single, consistent framework.

Usage:
    from app.coordination.resilience_orchestrator import (
        ResilienceOrchestrator,
        ResilienceConfig,
        RetryTier,
        get_resilience_orchestrator,
    )

    orchestrator = get_resilience_orchestrator()

    # Execute with automatic resilience
    result = await orchestrator.execute_with_resilience(
        operation=lambda: fetch_data(),
        config=ResilienceConfig(tier=RetryTier.STANDARD),
        operation_name="fetch_data",
    )

    # With circuit breaker for specific target
    result = await orchestrator.execute_with_resilience(
        operation=lambda: sync_to_node(node_id),
        config=ResilienceConfig(
            tier=RetryTier.PATIENT,
            circuit_breaker_target=node_id,
        ),
        operation_name="sync_to_node",
    )

January 2026: Created as part of long-term stability improvements.
Expected impact: ~800 LOC saved, consistent retry behavior, better metrics.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

from app.utils.retry import RetryConfig

logger = logging.getLogger(__name__)

__all__ = [
    "RetryTier",
    "ResilienceConfig",
    "ResilienceOrchestrator",
    "CircuitOpenError",
    "RetryExhaustedError",
    "OperationTimeoutError",
    "get_resilience_orchestrator",
    "reset_resilience_orchestrator",
    # Pre-configured configs for common patterns
    "RESILIENCE_QUICK",
    "RESILIENCE_STANDARD",
    "RESILIENCE_PATIENT",
    "RESILIENCE_PERSISTENT",
]

T = TypeVar("T")


# =============================================================================
# Exceptions
# =============================================================================

class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and operation cannot proceed."""

    def __init__(self, operation_name: str, target: str | None = None):
        self.operation_name = operation_name
        self.target = target
        message = f"Circuit open for {operation_name}"
        if target:
            message += f" (target: {target})"
        super().__init__(message)


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(
        self,
        operation_name: str,
        attempts: int,
        last_error: Exception | None = None,
    ):
        self.operation_name = operation_name
        self.attempts = attempts
        self.last_error = last_error
        message = f"{operation_name} failed after {attempts} attempts"
        if last_error:
            message += f": {last_error}"
        super().__init__(message)


class OperationTimeoutError(Exception):
    """Raised when operation exceeds timeout."""

    def __init__(self, operation_name: str, timeout_seconds: float):
        self.operation_name = operation_name
        self.timeout_seconds = timeout_seconds
        super().__init__(f"{operation_name} timed out after {timeout_seconds}s")


# =============================================================================
# Retry Tiers
# =============================================================================

class RetryTier(Enum):
    """Pre-defined retry behavior tiers.

    Use these tiers instead of custom retry configs to ensure consistent
    behavior across the codebase.
    """

    QUICK = "quick"
    """3 attempts, 1s base delay. For fast operations that should fail quickly."""

    STANDARD = "standard"
    """5 attempts, 5s base delay. Default for most operations."""

    PATIENT = "patient"
    """10 attempts, 30s base delay. For operations that may take time to recover."""

    PERSISTENT = "persistent"
    """Unlimited attempts, 60s base delay. For critical operations that must succeed."""


# Tier configurations
_TIER_CONFIGS: dict[RetryTier, RetryConfig] = {
    RetryTier.QUICK: RetryConfig(max_attempts=3, base_delay=1.0, max_delay=5.0),
    RetryTier.STANDARD: RetryConfig(max_attempts=5, base_delay=5.0, max_delay=60.0),
    RetryTier.PATIENT: RetryConfig(max_attempts=10, base_delay=30.0, max_delay=300.0),
    RetryTier.PERSISTENT: RetryConfig(
        max_attempts=0,  # 0 = unlimited
        base_delay=60.0,
        max_delay=600.0,
    ),
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ResilienceConfig:
    """Configuration for resilient operation execution.

    Attributes:
        tier: Retry tier defining attempt count and delays
        timeout_seconds: Per-attempt timeout (0 = no timeout)
        circuit_breaker_target: Target for circuit breaker tracking (e.g., node_id)
        emit_metrics: Whether to emit Prometheus metrics
        retryable_exceptions: Exception types that should trigger retry
        fatal_exceptions: Exception types that should immediately fail
    """

    tier: RetryTier = RetryTier.STANDARD
    timeout_seconds: float = 30.0
    circuit_breaker_target: str | None = None
    emit_metrics: bool = True

    # Exception handling
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default=(
            ConnectionError,
            TimeoutError,
            OSError,
            asyncio.TimeoutError,
        )
    )
    fatal_exceptions: tuple[type[Exception], ...] = field(
        default=(
            KeyboardInterrupt,
            SystemExit,
            MemoryError,
        )
    )

    @property
    def retry_config(self) -> RetryConfig:
        """Get the RetryConfig for this tier."""
        return _TIER_CONFIGS[self.tier]


# Pre-configured resilience configs for common patterns
RESILIENCE_QUICK = lambda: ResilienceConfig(tier=RetryTier.QUICK, timeout_seconds=10.0)
RESILIENCE_STANDARD = lambda: ResilienceConfig(tier=RetryTier.STANDARD, timeout_seconds=30.0)
RESILIENCE_PATIENT = lambda: ResilienceConfig(tier=RetryTier.PATIENT, timeout_seconds=120.0)
RESILIENCE_PERSISTENT = lambda: ResilienceConfig(tier=RetryTier.PERSISTENT, timeout_seconds=300.0)


# =============================================================================
# Resilience Orchestrator
# =============================================================================

@dataclass
class ResilienceStats:
    """Statistics for resilience operations."""

    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_retries: int = 0
    circuit_opens: int = 0
    timeouts: int = 0
    last_operation_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations


class ResilienceOrchestrator:
    """Unified resilience coordination for all operations.

    Provides:
    - Retry with exponential backoff
    - Circuit breaker integration
    - Per-operation timeouts
    - Metrics emission
    - Consistent behavior across all coordination operations
    """

    _instance: ResilienceOrchestrator | None = None

    def __init__(self):
        self._circuit_breaker: Any | None = None
        self._stats = ResilienceStats()
        self._operation_stats: dict[str, ResilienceStats] = {}

    @classmethod
    def get_instance(cls) -> ResilienceOrchestrator:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    @property
    def circuit_breaker(self) -> Any:
        """Lazy-load circuit breaker to avoid import cycles."""
        if self._circuit_breaker is None:
            try:
                from app.distributed.circuit_breaker import CircuitBreaker

                self._circuit_breaker = CircuitBreaker(
                    failure_threshold=5,
                    recovery_timeout=60.0,
                    operation_type="resilience",
                )
            except ImportError:
                logger.warning(
                    "[ResilienceOrchestrator] CircuitBreaker not available, "
                    "circuit protection disabled"
                )
        return self._circuit_breaker

    async def execute_with_resilience(
        self,
        operation: Callable[[], T] | Callable[[], asyncio.Future[T]],
        config: ResilienceConfig,
        operation_name: str,
    ) -> T:
        """Execute operation with retry, circuit breaker, and timeout.

        Args:
            operation: Callable that returns result or awaitable
            config: Resilience configuration
            operation_name: Name for logging and metrics

        Returns:
            Result from the operation

        Raises:
            CircuitOpenError: If circuit breaker is open
            RetryExhaustedError: If all retries exhausted
            OperationTimeoutError: If operation times out
        """
        target = config.circuit_breaker_target or "default"
        retry_config = config.retry_config

        # Check circuit breaker
        if self.circuit_breaker and config.circuit_breaker_target:
            if not self.circuit_breaker.can_execute(target):
                self._stats.circuit_opens += 1
                raise CircuitOpenError(operation_name, target)

        # Track stats
        self._stats.total_operations += 1
        op_stats = self._get_operation_stats(operation_name)
        op_stats.total_operations += 1

        attempt = 0
        last_error: Exception | None = None
        max_attempts = retry_config.max_attempts or 1000  # 0 = unlimited → use high limit

        while attempt < max_attempts:
            attempt += 1

            try:
                # Execute with timeout
                result = await self._execute_with_timeout(
                    operation, config.timeout_seconds, operation_name
                )

                # Record success
                if self.circuit_breaker and config.circuit_breaker_target:
                    self.circuit_breaker.record_success(target)

                self._stats.successful_operations += 1
                op_stats.successful_operations += 1
                self._stats.last_operation_time = time.time()

                if attempt > 1:
                    logger.info(
                        f"[Resilience] {operation_name} succeeded after {attempt} attempts"
                    )

                return result

            except config.fatal_exceptions as e:
                # Fatal exceptions should not retry
                self._record_failure(op_stats, target, config, e)
                raise

            except config.retryable_exceptions as e:
                last_error = e
                self._stats.total_retries += 1
                op_stats.total_retries += 1

                # Record failure with circuit breaker
                if self.circuit_breaker and config.circuit_breaker_target:
                    self.circuit_breaker.record_failure(target, e)

                # Check if we should retry
                if retry_config.max_attempts == 0 or attempt < max_attempts:
                    delay = retry_config.get_delay(attempt)
                    logger.warning(
                        f"[Resilience] {operation_name} attempt {attempt} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    break

            except asyncio.TimeoutError as e:
                last_error = OperationTimeoutError(operation_name, config.timeout_seconds)
                self._stats.timeouts += 1
                op_stats.timeouts += 1

                if self.circuit_breaker and config.circuit_breaker_target:
                    self.circuit_breaker.record_failure(target, e)

                if retry_config.max_attempts == 0 or attempt < max_attempts:
                    delay = retry_config.get_delay(attempt)
                    logger.warning(
                        f"[Resilience] {operation_name} timed out after "
                        f"{config.timeout_seconds}s. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    break

            except Exception as e:
                # Unexpected exception - may or may not be retryable
                last_error = e
                logger.error(
                    f"[Resilience] {operation_name} unexpected error: {type(e).__name__}: {e}"
                )

                if self.circuit_breaker and config.circuit_breaker_target:
                    self.circuit_breaker.record_failure(target, e)

                # Retry unexpected exceptions too, but log them
                if retry_config.max_attempts == 0 or attempt < max_attempts:
                    delay = retry_config.get_delay(attempt)
                    await asyncio.sleep(delay)
                else:
                    break

        # All retries exhausted
        self._record_failure(op_stats, target, config, last_error)
        raise RetryExhaustedError(operation_name, attempt, last_error)

    async def _execute_with_timeout(
        self,
        operation: Callable[[], T] | Callable[[], asyncio.Future[T]],
        timeout_seconds: float,
        operation_name: str,
    ) -> T:
        """Execute operation with optional timeout."""
        # Call the operation
        result = operation()

        # If it's a coroutine, await it
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            if timeout_seconds > 0:
                return await asyncio.wait_for(result, timeout=timeout_seconds)
            return await result

        # Sync operation - return directly
        return result

    def _record_failure(
        self,
        op_stats: ResilienceStats,
        target: str,
        config: ResilienceConfig,
        error: Exception | None,
    ) -> None:
        """Record operation failure."""
        self._stats.failed_operations += 1
        op_stats.failed_operations += 1
        self._stats.last_operation_time = time.time()

    def _get_operation_stats(self, operation_name: str) -> ResilienceStats:
        """Get or create stats for an operation."""
        if operation_name not in self._operation_stats:
            self._operation_stats[operation_name] = ResilienceStats()
        return self._operation_stats[operation_name]

    def get_stats(self) -> dict[str, Any]:
        """Get current resilience statistics."""
        return {
            "total_operations": self._stats.total_operations,
            "successful_operations": self._stats.successful_operations,
            "failed_operations": self._stats.failed_operations,
            "success_rate": self._stats.success_rate,
            "total_retries": self._stats.total_retries,
            "circuit_opens": self._stats.circuit_opens,
            "timeouts": self._stats.timeouts,
            "last_operation_time": self._stats.last_operation_time,
            "per_operation_stats": {
                name: {
                    "total": stats.total_operations,
                    "successful": stats.successful_operations,
                    "failed": stats.failed_operations,
                    "retries": stats.total_retries,
                }
                for name, stats in self._operation_stats.items()
            },
        }

    def health_check(self) -> "HealthCheckResult":
        """Return health status for DaemonManager integration.

        Sprint 15 (Jan 3, 2026): Added for unified health monitoring.
        Sprint 15.4: Updated to return HealthCheckResult instead of dict.
        """
        from app.coordination.contracts import HealthCheckResult, CoordinatorStatus

        success_rate = self._stats.success_rate
        circuit_opens_recent = self._stats.circuit_opens

        # Determine health status based on success rate
        if success_rate < 0.5 or circuit_opens_recent > 10:
            coordinator_status = CoordinatorStatus.PAUSED
            healthy = False
        elif success_rate < 0.8 or circuit_opens_recent > 5:
            coordinator_status = CoordinatorStatus.RUNNING  # degraded but running
            healthy = True
        else:
            coordinator_status = CoordinatorStatus.RUNNING
            healthy = True

        return HealthCheckResult(
            healthy=healthy,
            status=coordinator_status,
            details={
                "total_operations": self._stats.total_operations,
                "success_rate": round(success_rate, 3),
                "failed_operations": self._stats.failed_operations,
                "total_retries": self._stats.total_retries,
                "circuit_opens": circuit_opens_recent,
                "timeouts": self._stats.timeouts,
                "last_operation_time": self._stats.last_operation_time,
                "tracked_operations": len(self._operation_stats),
            },
        )


# =============================================================================
# Singleton Access
# =============================================================================

def get_resilience_orchestrator() -> ResilienceOrchestrator:
    """Get the singleton ResilienceOrchestrator instance."""
    return ResilienceOrchestrator.get_instance()


def reset_resilience_orchestrator() -> None:
    """Reset the singleton (for testing)."""
    ResilienceOrchestrator.reset_instance()
