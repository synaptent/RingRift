"""Shared Resilience Utilities for Distributed Operations.

This module provides resilience patterns for tournament scripts
and other distributed operations, focusing on HOST-LEVEL HEALTH
tracking that goes beyond simple retry logic.

Core utilities (HostHealthTracker, RetryExecutor) track per-host
failure state and enable graceful degradation when hosts become
unreliable.

For simple retry logic (decorators, exponential backoff), use
app.utils.retry instead.

Usage:
    from scripts.lib.resilience import (
        HostHealthTracker,
        RetryExecutor,
        is_network_error,
        create_graceful_executor,
    )

    # Track host health
    tracker = HostHealthTracker(failure_threshold=3)
    if not tracker.is_degraded("host1"):
        try:
            do_work()
            tracker.record_success("host1")
        except NetworkError as e:
            if tracker.record_failure("host1", str(e)):
                logger.error("Host marked as degraded")

    # Retry with host health tracking
    executor = RetryExecutor(health_tracker=tracker)
    result = executor.execute("host1", lambda: ssh_command(), "sync data")

December 2025: Refactored to use app.utils.retry for basic retry logic.
This module focuses on host-level health tracking and graceful degradation.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, TypeVar

# Import retry utilities from canonical location
from app.utils.retry import RetryConfig as BaseRetryConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = [
    # Primary exports (unique to this module)
    "HostHealthTracker",
    "RetryExecutor",
    "RetryConfig",
    "is_network_error",
    "create_graceful_executor",
    # Backward-compat exports (use app.utils.retry instead)
    "exponential_backoff_delay",
    "retry_with_backoff",
]


@dataclass
class HostHealthTracker:
    """Track host health for resilient job scheduling.

    Monitors consecutive failures per host and supports marking hosts as
    degraded when they exceed failure thresholds. Degraded hosts can have
    their pending jobs reassigned to healthy hosts.

    Thread-safe for use in concurrent execution contexts.

    Example:
        tracker = HostHealthTracker(failure_threshold=3)

        # Check and update health
        if not tracker.is_degraded("host1"):
            try:
                result = execute_on_host("host1")
                tracker.record_success("host1")
            except NetworkError as e:
                became_degraded = tracker.record_failure("host1", str(e))
                if became_degraded:
                    reassign_jobs_from("host1")

        # Get healthy hosts for job assignment
        healthy = tracker.get_healthy_hosts(["host1", "host2", "host3"])
    """

    failure_threshold: int = 3
    _consecutive_failures: dict[str, int] = field(default_factory=dict)
    _degraded_hosts: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _failure_reasons: dict[str, str] = field(default_factory=dict)
    _total_failures: dict[str, int] = field(default_factory=dict)

    def record_success(self, host_name: str) -> None:
        """Record a successful operation on a host.

        Resets the consecutive failure counter for the host.
        Does NOT remove degraded status (use reset_host for that).
        """
        with self._lock:
            self._consecutive_failures[host_name] = 0

    def record_failure(self, host_name: str, reason: str = "") -> bool:
        """Record a failure on a host.

        Args:
            host_name: Name of the host that failed
            reason: Optional description of the failure

        Returns:
            True if this failure caused the host to become degraded
        """
        with self._lock:
            count = self._consecutive_failures.get(host_name, 0) + 1
            self._consecutive_failures[host_name] = count
            self._total_failures[host_name] = self._total_failures.get(host_name, 0) + 1

            if reason:
                self._failure_reasons[host_name] = reason

            if count >= self.failure_threshold and host_name not in self._degraded_hosts:
                self._degraded_hosts.add(host_name)
                logger.warning(
                    f"Host {host_name} marked as degraded after {count} consecutive failures"
                )
                return True
            return False

    def is_degraded(self, host_name: str) -> bool:
        """Check if a host is marked as degraded."""
        with self._lock:
            return host_name in self._degraded_hosts

    def reset_host(self, host_name: str) -> None:
        """Reset a host's health status (remove degraded status)."""
        with self._lock:
            self._consecutive_failures[host_name] = 0
            self._degraded_hosts.discard(host_name)
            self._failure_reasons.pop(host_name, None)

    def get_healthy_hosts(self, all_hosts: Iterable[str]) -> list[str]:
        """Get list of hosts that are not degraded."""
        with self._lock:
            return [h for h in all_hosts if h not in self._degraded_hosts]

    def get_degraded_hosts(self) -> set[str]:
        """Get set of all degraded hosts."""
        with self._lock:
            return set(self._degraded_hosts)

    def get_failure_count(self, host_name: str) -> int:
        """Get consecutive failure count for a host."""
        with self._lock:
            return self._consecutive_failures.get(host_name, 0)

    def get_failure_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary of all host failures.

        Returns:
            Dict mapping host name to status dict with keys:
            - consecutive_failures: Current consecutive failure count
            - total_failures: Total failures since tracker creation
            - degraded: Whether host is currently degraded
            - reason: Last failure reason (if any)
        """
        with self._lock:
            all_hosts = set(self._consecutive_failures.keys()) | self._degraded_hosts
            return {
                host: {
                    "consecutive_failures": self._consecutive_failures.get(host, 0),
                    "total_failures": self._total_failures.get(host, 0),
                    "degraded": host in self._degraded_hosts,
                    "reason": self._failure_reasons.get(host, ""),
                }
                for host in all_hosts
            }


@dataclass
class RetryConfig:
    """Configuration for retry behavior with SCP-specific settings.

    This is a standalone config class for backward compatibility.
    Uses app.utils.retry.RetryConfig internally for delay calculation.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    # SCP-specific settings
    scp_retries: int = 3
    scp_base_delay: float = 2.0

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt (0-indexed)."""
        config = BaseRetryConfig(
            max_attempts=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
        )
        return config.get_delay(attempt + 1)  # BaseRetryConfig uses 1-indexed


# Re-export commonly used functions from canonical location for backward compat
def exponential_backoff_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter_factor: float = 0.2,
) -> float:
    """Calculate exponential backoff delay with jitter.

    DEPRECATED: Use app.utils.retry.RetryConfig.get_delay() instead.

    Args:
        attempt: Zero-indexed attempt number
        base_delay: Base delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 30.0)
        jitter_factor: Random jitter as fraction of delay (default: 0.2)

    Returns:
        Delay in seconds with jitter applied
    """
    config = BaseRetryConfig(base_delay=base_delay, max_delay=max_delay, jitter=jitter_factor)
    return config.get_delay(attempt + 1)  # get_delay uses 1-indexed attempts


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_exceptions: tuple = (Exception,),
    on_retry: Callable[[int, Exception], None] | None = None,
    should_retry: Callable[[Exception], bool] | None = None,
) -> T:
    """Execute a function with exponential backoff retries.

    DEPRECATED: Use app.utils.retry.retry decorator or RetryConfig.attempts() instead.

    Args:
        func: Function to execute
        max_retries: Maximum number of attempts (default: 3)
        base_delay: Base delay between retries in seconds (default: 1.0)
        max_delay: Maximum delay between retries (default: 30.0)
        retry_exceptions: Tuple of exception types to retry on
        on_retry: Optional callback(attempt, exception) called before each retry
        should_retry: Optional predicate to decide if exception should be retried

    Returns:
        Result of successful function call

    Raises:
        Last exception if all retries fail
    """
    config = BaseRetryConfig(
        max_attempts=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )

    last_exception: Exception | None = None

    for attempt in config.attempts():
        try:
            return func()
        except retry_exceptions as e:
            last_exception = e

            # Check if we should retry this exception
            if should_retry is not None and not should_retry(e):
                raise

            if attempt.should_retry:
                if on_retry:
                    on_retry(attempt.number, e)
                attempt.wait()

    # All retries failed
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("retry_with_backoff failed without exception")


class RetryExecutor:
    """Execute operations with retry logic and host health tracking.

    Combines retry behavior with host health tracking for resilient
    distributed operations.

    Example:
        executor = RetryExecutor(
            health_tracker=tracker,
            config=RetryConfig(max_retries=3)
        )

        result = executor.execute(
            host="host1",
            operation=lambda: ssh_run("host1", cmd),
            operation_name="training job",
        )
    """

    def __init__(
        self,
        health_tracker: HostHealthTracker | None = None,
        config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
    ):
        self.health_tracker = health_tracker or HostHealthTracker()
        self.config = config or RetryConfig()
        self.logger = logger or logging.getLogger(__name__)

    def execute(
        self,
        host: str,
        operation: Callable[[], T],
        operation_name: str = "operation",
        is_network_error: Callable[[Exception], bool] | None = None,
    ) -> T:
        """Execute an operation with retries and health tracking.

        Args:
            host: Host name for health tracking
            operation: Function to execute
            operation_name: Description for logging
            is_network_error: Predicate to identify network errors (triggers degradation)

        Returns:
            Result of successful operation

        Raises:
            Exception from operation if all retries fail or host is degraded
        """
        if self.health_tracker.is_degraded(host):
            raise RuntimeError(f"Host {host} is degraded, skipping {operation_name}")

        last_exception: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                result = operation()
                self.health_tracker.record_success(host)
                return result
            except Exception as e:
                last_exception = e
                self.logger.warning(
                    f"[{host}] {operation_name} failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )

                # Check if this is a network error
                if is_network_error and is_network_error(e):
                    became_degraded = self.health_tracker.record_failure(host, str(e))
                    if became_degraded:
                        self.logger.error(f"[{host}] marked as degraded after network errors")
                        break

                # Backoff before retry
                if attempt < self.config.max_retries - 1:
                    delay = exponential_backoff_delay(
                        attempt,
                        self.config.base_delay,
                        self.config.max_delay,
                    )
                    self.logger.info(f"[{host}] Retrying {operation_name} in {delay:.1f}s...")
                    time.sleep(delay)

        if last_exception is not None:
            raise last_exception
        raise RuntimeError(f"{operation_name} failed on {host} without exception")


def is_network_error(error: Exception | str) -> bool:
    """Check if an error appears to be network-related.

    Args:
        error: Exception or error string to check

    Returns:
        True if error appears to be a network issue
    """
    error_str = str(error).lower()
    network_indicators = [
        "connection",
        "timeout",
        "timed out",
        "broken pipe",
        "unreachable",
        "refused",
        "reset by peer",
        "network",
        "ssh",
        "socket",
        "eof",
        "disconnect",
    ]
    return any(indicator in error_str for indicator in network_indicators)


def create_graceful_executor(
    worker_count: int,
    graceful_degradation: bool = True,
    logger: logging.Logger | None = None,
):
    """Create a ThreadPoolExecutor wrapper with graceful degradation.

    This is a factory that returns a context manager for executing
    tasks with automatic error handling and result collection.

    Args:
        worker_count: Number of concurrent workers
        graceful_degradation: If True, continue on failures
        logger: Logger for error messages

    Returns:
        GracefulExecutor instance
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    class GracefulExecutor:
        def __init__(self):
            self.pool = ThreadPoolExecutor(max_workers=worker_count)
            self.futures = {}
            self.results = []
            self.failures = []
            self._logger = logger or logging.getLogger(__name__)

        def submit(self, func: Callable, *args, task_id: str = "", **kwargs):
            """Submit a task for execution."""
            future = self.pool.submit(func, *args, **kwargs)
            self.futures[future] = task_id or str(len(self.futures))
            return future

        def collect_results(self) -> tuple[list, list[tuple[str, Exception]]]:
            """Collect all results, handling failures gracefully.

            Returns:
                Tuple of (successful_results, list of (task_id, exception) for failures)
            """
            for future in as_completed(self.futures):
                task_id = self.futures[future]
                try:
                    result = future.result()
                    self.results.append(result)
                except Exception as e:
                    self.failures.append((task_id, e))
                    if graceful_degradation:
                        self._logger.error(f"Task {task_id} failed: {e}")
                    else:
                        raise
            return self.results, self.failures

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.pool.shutdown(wait=True)
            return False

    return GracefulExecutor()
