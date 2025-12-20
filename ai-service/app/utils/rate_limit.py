"""Rate limiting utilities for controlling operation frequency.

This module provides simple rate limiting primitives for:
- Preventing API abuse
- Throttling expensive operations
- Implementing cooldown periods

Usage:
    from app.utils.rate_limit import RateLimiter, Cooldown, rate_limit

    # Token bucket rate limiter
    limiter = RateLimiter(rate=10, per_seconds=1.0)  # 10 ops/second
    if limiter.acquire():
        do_operation()

    # Simple cooldown
    cooldown = Cooldown(seconds=5.0)
    if cooldown.ready():
        cooldown.reset()
        do_expensive_operation()

    # Decorator
    @rate_limit(rate=5, per_seconds=60.0)
    def api_call():
        ...
"""

from __future__ import annotations

import asyncio
import functools
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class Cooldown:
    """Simple cooldown timer.

    Tracks time since last reset and reports when cooldown period has elapsed.

    Example:
        cooldown = Cooldown(seconds=60.0)

        def maybe_do_expensive_thing():
            if cooldown.ready():
                cooldown.reset()
                do_expensive_thing()
    """

    seconds: float
    _last_reset: float = field(default=0.0, init=False)

    def ready(self) -> bool:
        """Check if cooldown period has elapsed."""
        return time.time() - self._last_reset >= self.seconds

    def reset(self) -> None:
        """Reset the cooldown timer."""
        self._last_reset = time.time()

    def remaining(self) -> float:
        """Get seconds remaining until ready."""
        elapsed = time.time() - self._last_reset
        return max(0.0, self.seconds - elapsed)

    def elapsed(self) -> float:
        """Get seconds elapsed since last reset."""
        return time.time() - self._last_reset


class RateLimiter:
    """Token bucket rate limiter.

    Allows bursts up to bucket size, then limits to steady rate.

    Example:
        limiter = RateLimiter(rate=10, per_seconds=1.0)  # 10 ops/sec

        for item in items:
            if limiter.acquire():
                process(item)
            else:
                print("Rate limited, waiting...")
                limiter.wait()
                process(item)
    """

    def __init__(
        self,
        rate: float,
        per_seconds: float = 1.0,
        burst: float | None = None,
    ):
        """Initialize the rate limiter.

        Args:
            rate: Number of operations allowed
            per_seconds: Time period for the rate (default 1 second)
            burst: Maximum burst size (default: same as rate)
        """
        self.rate = rate
        self.per_seconds = per_seconds
        self.burst = burst if burst is not None else rate

        self._tokens = float(self.burst)
        self._last_update = time.time()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now

        # Add tokens for elapsed time
        tokens_to_add = elapsed * (self.rate / self.per_seconds)
        self._tokens = min(self.burst, self._tokens + tokens_to_add)

    def acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens.

        Args:
            tokens: Number of tokens to acquire (default 1)

        Returns:
            True if tokens were acquired, False if rate limited
        """
        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait(self, tokens: float = 1.0) -> None:
        """Wait until tokens are available, then acquire.

        Args:
            tokens: Number of tokens to acquire (default 1)
        """
        while not self.acquire(tokens):
            # Calculate wait time
            with self._lock:
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed * (self.per_seconds / self.rate)
            time.sleep(min(wait_time, 0.1))  # Check at least every 100ms

    async def wait_async(self, tokens: float = 1.0) -> None:
        """Async version of wait().

        Args:
            tokens: Number of tokens to acquire (default 1)
        """
        while not self.acquire(tokens):
            with self._lock:
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed * (self.per_seconds / self.rate)
            await asyncio.sleep(min(wait_time, 0.1))

    @property
    def available_tokens(self) -> float:
        """Get current number of available tokens."""
        with self._lock:
            self._refill()
            return self._tokens


class KeyedRateLimiter:
    """Rate limiter with separate limits per key.

    Useful for per-user or per-resource rate limiting.

    Example:
        limiter = KeyedRateLimiter(rate=10, per_seconds=60.0)  # 10/minute per key

        def handle_request(user_id: str):
            if not limiter.acquire(user_id):
                raise TooManyRequestsError()
            process_request()
    """

    def __init__(
        self,
        rate: float,
        per_seconds: float = 1.0,
        burst: float | None = None,
    ):
        """Initialize the keyed rate limiter.

        Args:
            rate: Number of operations allowed per key
            per_seconds: Time period for the rate
            burst: Maximum burst size per key
        """
        self.rate = rate
        self.per_seconds = per_seconds
        self.burst = burst

        self._limiters: dict[str, RateLimiter] = {}
        self._lock = threading.Lock()

    def _get_limiter(self, key: str) -> RateLimiter:
        """Get or create limiter for a key."""
        with self._lock:
            if key not in self._limiters:
                self._limiters[key] = RateLimiter(
                    rate=self.rate,
                    per_seconds=self.per_seconds,
                    burst=self.burst,
                )
            return self._limiters[key]

    def acquire(self, key: str, tokens: float = 1.0) -> bool:
        """Try to acquire tokens for a key.

        Args:
            key: Rate limit key (e.g., user ID)
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False if rate limited
        """
        return self._get_limiter(key).acquire(tokens)

    def wait(self, key: str, tokens: float = 1.0) -> None:
        """Wait until tokens are available for a key."""
        self._get_limiter(key).wait(tokens)

    async def wait_async(self, key: str, tokens: float = 1.0) -> None:
        """Async wait until tokens are available for a key."""
        await self._get_limiter(key).wait_async(tokens)

    def cleanup(self, max_age_seconds: float = 3600.0) -> int:
        """Remove stale limiters that haven't been used recently.

        Args:
            max_age_seconds: Remove limiters inactive for this long

        Returns:
            Number of limiters removed
        """
        cutoff = time.time() - max_age_seconds
        removed = 0

        with self._lock:
            keys_to_remove = [
                key
                for key, limiter in self._limiters.items()
                if limiter._last_update < cutoff
            ]
            for key in keys_to_remove:
                del self._limiters[key]
                removed += 1

        return removed


def rate_limit(
    rate: float,
    per_seconds: float = 1.0,
    burst: float | None = None,
    wait: bool = True,
) -> Callable[[F], F]:
    """Decorator to rate limit a function.

    Args:
        rate: Number of calls allowed
        per_seconds: Time period for the rate
        burst: Maximum burst size
        wait: If True, wait when rate limited. If False, raise exception.

    Returns:
        Decorated function

    Example:
        @rate_limit(rate=10, per_seconds=60.0)
        def send_notification(user_id: str):
            # At most 10 calls per minute
            ...
    """
    limiter = RateLimiter(rate=rate, per_seconds=per_seconds, burst=burst)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if wait:
                limiter.wait()
            elif not limiter.acquire():
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {func.__name__}"
                )
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def rate_limit_async(
    rate: float,
    per_seconds: float = 1.0,
    burst: float | None = None,
    wait: bool = True,
) -> Callable[[F], F]:
    """Async decorator to rate limit a function.

    Args:
        rate: Number of calls allowed
        per_seconds: Time period for the rate
        burst: Maximum burst size
        wait: If True, wait when rate limited. If False, raise exception.
    """
    limiter = RateLimiter(rate=rate, per_seconds=per_seconds, burst=burst)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if wait:
                await limiter.wait_async()
            elif not limiter.acquire():
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {func.__name__}"
                )
            return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded and wait=False."""


__all__ = [
    "Cooldown",
    "KeyedRateLimiter",
    "RateLimitExceeded",
    "RateLimiter",
    "rate_limit",
    "rate_limit_async",
]
