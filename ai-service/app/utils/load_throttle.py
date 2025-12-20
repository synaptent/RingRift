"""Load-based job throttling utilities.

This module provides functions to check system load and throttle job
spawning when the system is overloaded. This prevents runaway job
creation that can overwhelm cluster nodes.

Configuration via environment variables:
- RINGRIFT_MAX_LOAD_FACTOR: Max load relative to CPU count (default: 2.0)
- RINGRIFT_MAX_LOAD_ABSOLUTE: Absolute max load (default: 100.0)
- RINGRIFT_LOAD_BACKOFF_SECONDS: Wait interval when overloaded (default: 30)

Usage:
    from app.utils.load_throttle import is_system_overloaded, wait_for_load_decrease

    # Check before spawning a job
    if is_system_overloaded(verbose=True):
        if not wait_for_load_decrease(max_wait_seconds=300.0):
            print("System overloaded - skipping job")
            return

    # Or use async version
    if is_system_overloaded():
        if not await wait_for_load_decrease_async(max_wait_seconds=300.0):
            return

"""

from __future__ import annotations

import asyncio
import os
import time

__all__ = [
    "LOAD_BACKOFF_SECONDS",
    "MAX_LOAD_ABSOLUTE",
    # Configuration
    "MAX_LOAD_FACTOR",
    "LoadThrottler",
    "get_cpu_count",
    "get_load_info",
    "get_system_load",
    "is_system_overloaded",
    "wait_for_load_decrease",
    "wait_for_load_decrease_async",
]

# Configuration from environment
MAX_LOAD_FACTOR = float(os.environ.get("RINGRIFT_MAX_LOAD_FACTOR", "2.0"))
MAX_LOAD_ABSOLUTE = float(os.environ.get("RINGRIFT_MAX_LOAD_ABSOLUTE", "100.0"))
LOAD_BACKOFF_SECONDS = float(os.environ.get("RINGRIFT_LOAD_BACKOFF_SECONDS", "30.0"))


def get_system_load() -> tuple[float, float, float]:
    """Get system load averages (1min, 5min, 15min).

    Returns (0.0, 0.0, 0.0) on systems without getloadavg (e.g., Windows).
    """
    try:
        return os.getloadavg()
    except (OSError, AttributeError):
        return (0.0, 0.0, 0.0)


def get_cpu_count() -> int:
    """Get number of CPU cores."""
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def is_system_overloaded(verbose: bool = False) -> bool:
    """Check if system is too overloaded to spawn new jobs.

    Returns True if:
    - Load average exceeds MAX_LOAD_FACTOR * cpu_count, OR
    - Load average exceeds MAX_LOAD_ABSOLUTE

    Uses 5-minute load average to avoid reacting to brief spikes.

    Args:
        verbose: If True, print a message when overloaded

    Returns:
        True if system is overloaded, False otherwise
    """
    _, load_5min, _ = get_system_load()
    cpu_count = get_cpu_count()

    relative_threshold = MAX_LOAD_FACTOR * cpu_count
    threshold = min(relative_threshold, MAX_LOAD_ABSOLUTE)

    is_overloaded = load_5min > threshold

    if is_overloaded and verbose:
        print(f"[load-throttle] System overloaded: load={load_5min:.1f}, "
              f"threshold={threshold:.1f} (CPUs={cpu_count}, factor={MAX_LOAD_FACTOR})")

    return is_overloaded


def get_load_info() -> dict:
    """Get detailed load information.

    Returns:
        Dictionary with load averages, CPU count, thresholds, and overload status
    """
    load_1min, load_5min, load_15min = get_system_load()
    cpu_count = get_cpu_count()
    relative_threshold = MAX_LOAD_FACTOR * cpu_count
    threshold = min(relative_threshold, MAX_LOAD_ABSOLUTE)

    return {
        "load_1min": load_1min,
        "load_5min": load_5min,
        "load_15min": load_15min,
        "cpu_count": cpu_count,
        "max_load_factor": MAX_LOAD_FACTOR,
        "max_load_absolute": MAX_LOAD_ABSOLUTE,
        "effective_threshold": threshold,
        "is_overloaded": load_5min > threshold,
    }


def wait_for_load_decrease(
    max_wait_seconds: float = 300.0,
    verbose: bool = True,
    check_interval: float | None = None,
) -> bool:
    """Wait synchronously for system load to decrease.

    Args:
        max_wait_seconds: Maximum time to wait (default 5 minutes)
        verbose: If True, print progress messages
        check_interval: Seconds between checks (default: LOAD_BACKOFF_SECONDS)

    Returns:
        True if load decreased within timeout, False otherwise
    """
    if check_interval is None:
        check_interval = LOAD_BACKOFF_SECONDS

    start_time = time.time()

    while is_system_overloaded(verbose=False):
        elapsed = time.time() - start_time
        if elapsed >= max_wait_seconds:
            if verbose:
                _, load_5min, _ = get_system_load()
                print(f"[load-throttle] Load still high after {elapsed:.0f}s wait "
                      f"(load={load_5min:.1f}). Giving up.")
            return False

        if verbose:
            _, load_5min, _ = get_system_load()
            print(f"[load-throttle] Waiting for load to decrease "
                  f"(current={load_5min:.1f}, waited={elapsed:.0f}s)...")

        time.sleep(check_interval)

    return True


async def wait_for_load_decrease_async(
    max_wait_seconds: float = 300.0,
    verbose: bool = True,
    check_interval: float | None = None,
) -> bool:
    """Wait asynchronously for system load to decrease.

    Args:
        max_wait_seconds: Maximum time to wait (default 5 minutes)
        verbose: If True, print progress messages
        check_interval: Seconds between checks (default: LOAD_BACKOFF_SECONDS)

    Returns:
        True if load decreased within timeout, False otherwise
    """
    if check_interval is None:
        check_interval = LOAD_BACKOFF_SECONDS

    start_time = time.time()

    while is_system_overloaded(verbose=False):
        elapsed = time.time() - start_time
        if elapsed >= max_wait_seconds:
            if verbose:
                _, load_5min, _ = get_system_load()
                print(f"[load-throttle] Load still high after {elapsed:.0f}s wait "
                      f"(load={load_5min:.1f}). Giving up.")
            return False

        if verbose:
            _, load_5min, _ = get_system_load()
            print(f"[load-throttle] Waiting for load to decrease "
                  f"(current={load_5min:.1f}, waited={elapsed:.0f}s)...")

        await asyncio.sleep(check_interval)

    return True


class LoadThrottler:
    """Context manager for load-aware job execution.

    Usage:
        throttler = LoadThrottler(max_wait_seconds=300.0)

        # Synchronous
        with throttler:
            run_expensive_job()

        # Async
        async with throttler:
            await run_expensive_job_async()
    """

    def __init__(
        self,
        max_wait_seconds: float = 300.0,
        verbose: bool = True,
        skip_if_overloaded: bool = True,
    ):
        """Initialize the throttler.

        Args:
            max_wait_seconds: Maximum time to wait for load to decrease
            verbose: If True, print progress messages
            skip_if_overloaded: If True, skip execution if still overloaded after waiting
        """
        self.max_wait_seconds = max_wait_seconds
        self.verbose = verbose
        self.skip_if_overloaded = skip_if_overloaded
        self._should_skip = False

    @property
    def should_skip(self) -> bool:
        """Check if execution was skipped due to overload."""
        return self._should_skip

    def __enter__(self) -> LoadThrottler:
        if is_system_overloaded(verbose=self.verbose) and not wait_for_load_decrease(
            max_wait_seconds=self.max_wait_seconds,
            verbose=self.verbose,
        ):
            if self.skip_if_overloaded:
                self._should_skip = True
                return self
            raise RuntimeError("System overloaded - cannot proceed")
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> None:
        pass

    async def __aenter__(self) -> LoadThrottler:
        if is_system_overloaded(verbose=self.verbose) and not await wait_for_load_decrease_async(
            max_wait_seconds=self.max_wait_seconds,
            verbose=self.verbose,
        ):
            if self.skip_if_overloaded:
                self._should_skip = True
                return self
            raise RuntimeError("System overloaded - cannot proceed")
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> None:
        pass
