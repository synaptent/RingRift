"""Handler Timeout Decorator for P2P HTTP Handlers.

Provides timeout protection for HTTP handlers to prevent stuck handlers from
blocking P2P communication and causing cluster partitioning.

January 2026: Created as part of P2P critical hardening (Phase 1).
Problem: HTTP handlers without timeouts can block indefinitely on slow operations,
preventing all P2P communication to a node and causing cascade failures.

Usage:
    from scripts.p2p.handlers.timeout_decorator import handler_timeout

    class MyHandlersMixin:
        @handler_timeout(30)  # 30 second timeout
        async def handle_my_endpoint(self, request: web.Request) -> web.Response:
            # Handler implementation
            pass

    # With custom error response
    @handler_timeout(60, error_message="Operation timed out")
    async def handle_slow_operation(self, request: web.Request) -> web.Response:
        pass

Timeout Recommendations by Handler Type:
    - Gossip/heartbeat handlers: 30s (fast, critical for cluster health)
    - Tournament/job handlers: 60s (moderate complexity)
    - Delivery/sync handlers: 120s (may involve file I/O)
    - Admin/debug handlers: 300s (manual operations, less critical)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from aiohttp import web

logger = logging.getLogger(__name__)

# Type variable for handler return type
T = TypeVar("T")

# January 2026: Use centralized timeouts from loop_constants where applicable
try:
    from scripts.p2p.loops.loop_constants import LoopTimeouts
    DEFAULT_HANDLER_TIMEOUT = LoopTimeouts.HTTP_LONG  # 30.0 seconds default
    HANDLER_TIMEOUT_GOSSIP = LoopTimeouts.HTTP_LONG  # 30.0 - critical for cluster
    HANDLER_TIMEOUT_TOURNAMENT = 60.0  # Not in LoopTimeouts - handler-specific
    HANDLER_TIMEOUT_DELIVERY = LoopTimeouts.SYNC_LOCK  # 120.0 - may involve file I/O
    HANDLER_TIMEOUT_ADMIN = LoopTimeouts.SYNC_OPERATION  # 300.0 - manual ops
except ImportError:
    # Fallback values matching LoopTimeouts defaults
    DEFAULT_HANDLER_TIMEOUT = 30.0
    HANDLER_TIMEOUT_GOSSIP = 30.0
    HANDLER_TIMEOUT_TOURNAMENT = 60.0
    HANDLER_TIMEOUT_DELIVERY = 120.0
    HANDLER_TIMEOUT_ADMIN = 300.0


def handler_timeout(
    timeout_seconds: float = DEFAULT_HANDLER_TIMEOUT,
    error_message: str = "Handler timeout",
    log_timeout: bool = True,
    include_handler_name: bool = True,
) -> Callable:
    """Decorator to add timeout protection to P2P HTTP handlers.

    Wraps async handler methods with asyncio.wait_for() to enforce a maximum
    execution time. If the handler exceeds the timeout, returns a 503 Service
    Unavailable response instead of blocking indefinitely.

    Args:
        timeout_seconds: Maximum execution time in seconds (default: 30)
        error_message: Message to include in timeout response
        log_timeout: Whether to log timeout events (default: True)
        include_handler_name: Include handler name in error response (default: True)

    Returns:
        Decorated async handler function

    Example:
        @handler_timeout(30)
        async def handle_gossip(self, request: web.Request) -> web.Response:
            # This will timeout after 30 seconds
            ...

    Notes:
        - Only works with async handler methods
        - Returns 503 status code on timeout (service unavailable, retry later)
        - Logs timeout events for monitoring/debugging
        - Preserves original function metadata via functools.wraps
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Get handler name for logging/error messages
            handler_name = func.__name__

            # Extract 'self' for accessing node_id if available
            self_instance = args[0] if args else None
            node_id = getattr(self_instance, "node_id", "unknown")

            start_time = time.time()

            try:
                # Execute handler with timeout
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds,
                )
                return result

            except asyncio.TimeoutError:
                elapsed = time.time() - start_time

                if log_timeout:
                    logger.warning(
                        f"[handler_timeout] {handler_name} timed out after "
                        f"{elapsed:.1f}s (limit: {timeout_seconds}s) on node {node_id}"
                    )

                # Import here to avoid circular imports
                from aiohttp import web

                # Build error response
                error_body = {
                    "error": error_message,
                    "status": 503,
                    "timeout_seconds": timeout_seconds,
                    "elapsed_seconds": round(elapsed, 2),
                }

                if include_handler_name:
                    error_body["handler"] = handler_name

                return web.json_response(
                    error_body,
                    status=503,
                    headers={"X-Handler-Timeout": str(timeout_seconds)},
                )

            except asyncio.CancelledError:
                # Don't catch cancellation - let it propagate
                elapsed = time.time() - start_time
                logger.debug(
                    f"[handler_timeout] {handler_name} cancelled after {elapsed:.1f}s"
                )
                raise

        return wrapper

    return decorator


def handler_timeout_with_fallback(
    timeout_seconds: float = DEFAULT_HANDLER_TIMEOUT,
    fallback_response: Callable[..., "web.Response"] | None = None,
) -> Callable:
    """Decorator with custom fallback response on timeout.

    Similar to handler_timeout but allows specifying a custom fallback
    response function that can access the request context.

    Args:
        timeout_seconds: Maximum execution time in seconds
        fallback_response: Callable that receives (request, handler_name, elapsed)
                          and returns a web.Response

    Example:
        def my_fallback(request, handler_name, elapsed):
            return web.json_response({"cached": True, "stale": True}, status=200)

        @handler_timeout_with_fallback(30, fallback_response=my_fallback)
        async def handle_data(self, request: web.Request) -> web.Response:
            # On timeout, returns cached data instead of error
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            handler_name = func.__name__
            start_time = time.time()

            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds,
                )

            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                logger.warning(
                    f"[handler_timeout] {handler_name} timed out after {elapsed:.1f}s"
                )

                if fallback_response:
                    # Try to get request from args
                    # Handler signature is typically (self, request) or (request,)
                    request = None
                    for arg in args:
                        if hasattr(arg, "method") and hasattr(arg, "path"):
                            request = arg
                            break

                    try:
                        return fallback_response(request, handler_name, elapsed)
                    except Exception as e:
                        logger.error(f"Fallback response failed: {e}")

                # Default error response
                from aiohttp import web

                return web.json_response(
                    {
                        "error": "Handler timeout",
                        "status": 503,
                        "handler": handler_name,
                    },
                    status=503,
                )

            except asyncio.CancelledError:
                raise

        return wrapper

    return decorator


class HandlerTimeoutMiddleware:
    """AIOHTTP middleware for global handler timeout enforcement.

    Alternative to per-handler decorators. Applies a default timeout
    to all handlers, with per-route overrides via route configuration.

    Usage:
        app = web.Application(middlewares=[
            HandlerTimeoutMiddleware(default_timeout=60).middleware,
        ])

        # Route-specific override
        app.router.add_post("/gossip", handle_gossip, timeout=30)
    """

    def __init__(
        self,
        default_timeout: float = 60.0,
        route_timeouts: dict[str, float] | None = None,
    ):
        """Initialize middleware.

        Args:
            default_timeout: Default timeout for all handlers
            route_timeouts: Dict mapping route paths to specific timeouts
        """
        self.default_timeout = default_timeout
        self.route_timeouts = route_timeouts or {}

    @property
    def middleware(self):
        """Return the middleware handler function."""

        async def timeout_middleware(app, handler):
            async def middleware_handler(request):
                # Get timeout for this route
                path = request.path
                timeout = self.route_timeouts.get(path, self.default_timeout)

                try:
                    return await asyncio.wait_for(
                        handler(request),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    from aiohttp import web

                    logger.warning(
                        f"[middleware] Handler for {path} timed out after {timeout}s"
                    )
                    return web.json_response(
                        {
                            "error": "Request timeout",
                            "status": 503,
                            "path": path,
                            "timeout_seconds": timeout,
                        },
                        status=503,
                    )

            return middleware_handler

        return timeout_middleware


__all__ = [
    "handler_timeout",
    "handler_timeout_with_fallback",
    "HandlerTimeoutMiddleware",
    # Timeout constants
    "DEFAULT_HANDLER_TIMEOUT",
    "HANDLER_TIMEOUT_GOSSIP",
    "HANDLER_TIMEOUT_TOURNAMENT",
    "HANDLER_TIMEOUT_DELIVERY",
    "HANDLER_TIMEOUT_ADMIN",
]
