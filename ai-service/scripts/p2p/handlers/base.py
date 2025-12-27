"""Base class for P2P HTTP Handler Mixins.

Provides common utilities for response formatting, error handling, authentication,
and body parsing. All handler mixins should inherit from BaseP2PHandler.

December 2025: Created as part of handler consolidation to reduce code duplication.
Expected savings: ~745 LOC across 14 handler files through shared utilities.

Usage:
    from scripts.p2p.handlers.base import BaseP2PHandler

    class MyHandlersMixin(BaseP2PHandler):
        async def handle_my_endpoint(self, request: web.Request) -> web.Response:
            # Check auth
            if not await self.check_auth(request):
                return self.auth_error()

            # Parse JSON body
            body = await self.parse_json_body(request)
            if body is None:
                return self.error_response("Invalid JSON body", status=400)

            # Process request...
            return self.json_response({"status": "ok", "result": result})

Common Patterns Consolidated:
    - Response formatting (117 instances -> 1 method)
    - Error handling (51 instances -> 3 methods)
    - Auth checks (10 instances -> 1 method)
    - JSON body parsing (40+ instances -> 2 methods)
    - Gzip detection and decompression (6 instances -> 1 method)
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from abc import ABC
from typing import TYPE_CHECKING, Any, Protocol

from aiohttp import web

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class P2PHandlerProtocol(Protocol):
    """Protocol defining required attributes for P2P handlers.

    Any class mixing in P2P handlers must provide these attributes.
    """

    node_id: str
    auth_token: str | None
    leader_id: str | None

    def _is_request_authorized(self, request: web.Request) -> bool:
        """Check if request is authorized."""
        ...


class BaseP2PHandler(ABC):
    """Base class for P2P HTTP handler mixins.

    Provides common utilities for:
    - Response formatting (json_response, error_response)
    - Authentication checking (check_auth, auth_error)
    - Body parsing (parse_json_body, parse_gzip_body)
    - Error responses (error_response, not_found, bad_request)

    Requires the implementing class to have:
    - node_id: str
    - auth_token: str | None
    - _is_request_authorized(request) method

    Note: This is designed as an abstract base for handler mixins.
    Handler mixins can directly inherit from this or use its methods via composition.
    """

    # Type hints for required attributes (must be provided by implementing class)
    node_id: str
    auth_token: str | None

    # ==========================================================================
    # Response Formatting
    # ==========================================================================

    def json_response(
        self,
        data: dict[str, Any],
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> web.Response:
        """Create a JSON response with consistent formatting.

        Args:
            data: Response data dictionary
            status: HTTP status code (default: 200)
            headers: Additional response headers

        Returns:
            aiohttp web.Response with JSON content
        """
        response_headers = {"X-Node-ID": getattr(self, "node_id", "unknown")}
        if headers:
            response_headers.update(headers)

        return web.json_response(
            data,
            status=status,
            headers=response_headers,
        )

    def error_response(
        self,
        message: str,
        status: int = 500,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> web.Response:
        """Create an error response with consistent formatting.

        Args:
            message: Human-readable error message
            status: HTTP status code (default: 500)
            error_code: Machine-readable error code (optional)
            details: Additional error details (optional)

        Returns:
            aiohttp web.Response with error JSON
        """
        error_data: dict[str, Any] = {
            "error": message,
            "status": status,
            "timestamp": time.time(),
        }

        if error_code:
            error_data["code"] = error_code

        # Keep a nested details object for structured clients, but also provide a
        # shallow top-level projection for backward compatibility with older
        # handlers/tests that expected fields like "acquired"/"released" at the
        # top level.
        if details:
            error_data["details"] = details
            for k, v in details.items():
                error_data.setdefault(k, v)

        return self.json_response(error_data, status=status)

    def not_found(self, resource: str = "Resource") -> web.Response:
        """Return 404 Not Found response.

        Args:
            resource: Name of the resource not found

        Returns:
            404 error response
        """
        return self.error_response(
            f"{resource} not found",
            status=404,
            error_code="NOT_FOUND",
        )

    def bad_request(self, message: str = "Bad request") -> web.Response:
        """Return 400 Bad Request response.

        Args:
            message: Error message describing what was wrong

        Returns:
            400 error response
        """
        return self.error_response(
            message,
            status=400,
            error_code="BAD_REQUEST",
        )

    # ==========================================================================
    # Authentication
    # ==========================================================================

    def check_auth(self, request: web.Request) -> bool:
        """Check if request is authorized.

        Uses the implementing class's _is_request_authorized method if available,
        or falls back to checking auth_token directly.

        Args:
            request: aiohttp web.Request

        Returns:
            True if authorized, False otherwise
        """
        # If no auth token configured, all requests are authorized
        if not getattr(self, "auth_token", None):
            return True

        # Use implementing class's method if available
        if hasattr(self, "_is_request_authorized"):
            return self._is_request_authorized(request)

        # Fallback: check Authorization header directly
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            return token == self.auth_token

        # Also check X-Auth-Token header
        token = request.headers.get("X-Auth-Token", "")
        return token == self.auth_token

    def auth_error(self) -> web.Response:
        """Return 401 Unauthorized response.

        Returns:
            401 error response with consistent formatting
        """
        return self.error_response(
            "Unauthorized: Invalid or missing authentication",
            status=401,
            error_code="UNAUTHORIZED",
        )

    # ==========================================================================
    # Body Parsing
    # ==========================================================================

    async def parse_json_body(self, request: web.Request) -> dict[str, Any] | None:
        """Parse JSON body from request.

        Args:
            request: aiohttp web.Request

        Returns:
            Parsed JSON dict, or None if parsing failed
        """
        try:
            return await request.json()
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"[{self.__class__.__name__}] JSON parse failed: {e}")
            return None

    async def parse_gzip_body(self, request: web.Request) -> dict[str, Any] | None:
        """Parse body that may be gzip-compressed.

        Uses magic byte detection (0x1f 0x8b) to determine if content is
        actually gzipped, handling clients that set Content-Encoding: gzip
        but send raw JSON.

        Args:
            request: aiohttp web.Request

        Returns:
            Parsed JSON dict, or None if parsing failed
        """
        try:
            raw_data = await request.read()

            # Check for gzip magic bytes
            if len(raw_data) >= 2 and raw_data[0] == 0x1F and raw_data[1] == 0x8B:
                # Actually gzipped - decompress
                try:
                    decompressed = gzip.decompress(raw_data)
                    return json.loads(decompressed)
                except (gzip.BadGzipFile, OSError, json.JSONDecodeError) as e:
                    logger.debug(f"[{self.__class__.__name__}] Gzip decompress failed: {e}")
                    return None
            else:
                # Not actually gzipped, try to parse as JSON directly
                # This handles clients that set Content-Encoding: gzip but send raw JSON
                try:
                    return json.loads(raw_data)
                except json.JSONDecodeError:
                    # Last resort: try parsing as UTF-8 text
                    try:
                        return json.loads(raw_data.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logger.debug(f"[{self.__class__.__name__}] JSON parse failed: {e}")
                        return None

        except (OSError, MemoryError) as e:
            logger.warning(f"[{self.__class__.__name__}] Body read failed: {e}")
            return None

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    def get_client_ip(self, request: web.Request) -> str:
        """Get client IP address from request.

        Checks X-Forwarded-For header for proxied requests,
        falls back to direct connection IP.

        Args:
            request: aiohttp web.Request

        Returns:
            Client IP address string
        """
        # Check for forwarded header
        forwarded_for = request.headers.get("X-Forwarded-For", "")
        if forwarded_for:
            # Take first IP in chain (original client)
            return forwarded_for.split(",")[0].strip()

        # Fall back to direct connection
        if request.transport:
            peername = request.transport.get_extra_info("peername")
            if peername:
                return peername[0]

        return "unknown"

    def log_request(
        self,
        request: web.Request,
        message: str = "",
        level: int = logging.DEBUG,
    ) -> None:
        """Log request with consistent formatting.

        Args:
            request: aiohttp web.Request
            message: Additional message to log
            level: Logging level (default: DEBUG)
        """
        client_ip = self.get_client_ip(request)
        method = request.method
        path = request.path

        log_msg = f"[{self.__class__.__name__}] {method} {path} from {client_ip}"
        if message:
            log_msg += f" - {message}"

        logger.log(level, log_msg)


# =============================================================================
# Module-level utilities for handlers that don't inherit from BaseP2PHandler
# =============================================================================

def make_json_response(
    data: dict[str, Any],
    status: int = 200,
    node_id: str = "unknown",
) -> web.Response:
    """Create a JSON response (standalone function).

    For handlers that don't inherit from BaseP2PHandler.

    Args:
        data: Response data dictionary
        status: HTTP status code
        node_id: Node ID for X-Node-ID header

    Returns:
        aiohttp web.Response
    """
    return web.json_response(
        data,
        status=status,
        headers={"X-Node-ID": node_id},
    )


def make_error_response(
    message: str,
    status: int = 500,
    node_id: str = "unknown",
) -> web.Response:
    """Create an error response (standalone function).

    For handlers that don't inherit from BaseP2PHandler.

    Args:
        message: Error message
        status: HTTP status code
        node_id: Node ID for X-Node-ID header

    Returns:
        aiohttp web.Response
    """
    return web.json_response(
        {
            "error": message,
            "status": status,
            "timestamp": time.time(),
        },
        status=status,
        headers={"X-Node-ID": node_id},
    )


__all__ = [
    "BaseP2PHandler",
    "P2PHandlerProtocol",
    "make_json_response",
    "make_error_response",
]
