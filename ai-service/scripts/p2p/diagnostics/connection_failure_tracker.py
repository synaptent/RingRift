"""Categorize and track connection failures by type."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger("p2p.diagnostics.connection")


class FailureType(Enum):
    """Categories of connection failures."""

    TIMEOUT = "timeout"
    REFUSED = "refused"
    DNS_FAILURE = "dns_failure"
    TLS_ERROR = "tls_error"
    RESET = "reset"
    POOL_EXHAUSTED = "pool_exhausted"
    CIRCUIT_OPEN = "circuit_open"
    HOST_UNREACHABLE = "host_unreachable"
    UNKNOWN = "unknown"


@dataclass
class ConnectionFailure:
    """Record of a connection failure."""

    node_id: str
    failure_type: FailureType
    transport: str  # "tailscale", "ssh", "http", "relay"
    timestamp: float
    error_msg: str
    host: str = ""
    port: int = 0


class ConnectionFailureTracker:
    """Track connection failures for pattern detection.

    Features:
    - Categorizes failures by type (timeout, refused, etc.)
    - Tracks by transport (tailscale, ssh, http)
    - Identifies worst-offending nodes
    - Provides pattern detection for systemic issues
    """

    def __init__(self, window: float = 600.0, max_failures: int = 5000) -> None:
        """Initialize the tracker.

        Args:
            window: Time window in seconds for analysis (default 10 min)
            max_failures: Maximum failures to keep in memory
        """
        self._failures: list[ConnectionFailure] = []
        self._window = window
        self._max_failures = max_failures

    def record_failure(
        self,
        node_id: str,
        error: Exception | str,
        transport: str = "unknown",
        host: str = "",
        port: int = 0,
    ) -> FailureType:
        """Record a connection failure and return its categorized type.

        Args:
            node_id: The target peer's node ID
            error: The exception or error message
            transport: Transport type (tailscale, ssh, http, relay)
            host: Target host (optional, for detailed logging)
            port: Target port (optional)

        Returns:
            The categorized FailureType
        """
        error_str = str(error)
        failure_type = self._categorize_error(error)

        failure = ConnectionFailure(
            node_id=node_id,
            failure_type=failure_type,
            transport=transport,
            timestamp=time.time(),
            error_msg=error_str[:200],
            host=host,
            port=port,
        )
        self._failures.append(failure)
        self._prune_old()

        # Log with structured format for grep-ability
        logger.warning(
            f"CONN_FAIL: node={node_id[:20]} type={failure_type.value} "
            f"transport={transport} host={host}:{port} error={error_str[:80]}"
        )

        return failure_type

    def _categorize_error(self, error: Exception | str) -> FailureType:
        """Categorize an exception or error string into a failure type."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower() if isinstance(error, Exception) else ""

        # Timeout patterns
        if any(
            p in error_str
            for p in ["timeout", "timed out", "deadline exceeded", "asyncio.timeout"]
        ):
            return FailureType.TIMEOUT

        # Connection refused
        if any(p in error_str for p in ["refused", "errno 111", "errno 61"]):
            return FailureType.REFUSED

        # Connection reset
        if any(p in error_str for p in ["reset", "errno 104", "errno 54"]):
            return FailureType.RESET

        # DNS failures
        if any(
            p in error_str
            for p in ["dns", "name resolution", "getaddrinfo", "nodename nor servname"]
        ):
            return FailureType.DNS_FAILURE

        # TLS/SSL errors
        if any(p in error_str for p in ["ssl", "tls", "certificate", "handshake"]):
            return FailureType.TLS_ERROR

        # Pool exhaustion
        if any(p in error_str for p in ["pool", "exhausted", "too many connections"]):
            return FailureType.POOL_EXHAUSTED

        # Circuit breaker
        if any(p in error_str for p in ["circuit", "breaker", "tripped"]):
            return FailureType.CIRCUIT_OPEN

        # Host unreachable
        if any(
            p in error_str for p in ["unreachable", "no route", "errno 113", "errno 65"]
        ):
            return FailureType.HOST_UNREACHABLE

        return FailureType.UNKNOWN

    def _prune_old(self) -> None:
        """Remove old failures outside the window."""
        cutoff = time.time() - self._window
        self._failures = [f for f in self._failures if f.timestamp > cutoff]

        # Also enforce max size
        if len(self._failures) > self._max_failures:
            self._failures = self._failures[-self._max_failures :]

    def get_failures_by_node(self, node_id: str) -> list[ConnectionFailure]:
        """Get all recent failures for a specific node."""
        self._prune_old()
        return [f for f in self._failures if f.node_id == node_id]

    def get_failure_rate(self, node_id: str | None = None) -> float:
        """Get failures per minute rate.

        Args:
            node_id: Filter to specific node (None for all)
        """
        self._prune_old()
        failures = self._failures
        if node_id:
            failures = [f for f in failures if f.node_id == node_id]

        if not failures:
            return 0.0

        window_minutes = self._window / 60.0
        return len(failures) / window_minutes

    def is_systemic_issue(self, failure_type: FailureType, threshold: int = 10) -> bool:
        """Check if a failure type indicates a systemic issue.

        Returns True if the same failure type affects multiple nodes.
        """
        self._prune_old()
        affected_nodes = set(
            f.node_id for f in self._failures if f.failure_type == failure_type
        )
        return len(affected_nodes) >= threshold

    def get_diagnostics(self) -> dict[str, Any]:
        """Return failure diagnostics.

        Returns dict with:
        - total_failures_10min: Count in window
        - by_type: Count per failure type
        - by_transport: Count per transport
        - worst_nodes: Top 5 failing nodes
        - systemic_issues: Failure types affecting 10+ nodes
        - failures_per_minute: Overall failure rate
        """
        self._prune_old()

        by_type: dict[str, int] = defaultdict(int)
        by_transport: dict[str, int] = defaultdict(int)
        by_node: dict[str, int] = defaultdict(int)

        for f in self._failures:
            by_type[f.failure_type.value] += 1
            by_transport[f.transport] += 1
            by_node[f.node_id] += 1

        # Find worst offenders
        worst_nodes = sorted(by_node.items(), key=lambda x: -x[1])[:5]
        worst_nodes = [(nid[:20], count) for nid, count in worst_nodes]

        # Check for systemic issues
        systemic = [
            ft.value
            for ft in FailureType
            if self.is_systemic_issue(ft, threshold=5)
        ]

        return {
            "total_failures_10min": len(self._failures),
            "by_type": dict(by_type),
            "by_transport": dict(by_transport),
            "worst_nodes": worst_nodes,
            "systemic_issues": systemic,
            "failures_per_minute": round(self.get_failure_rate(), 2),
        }

    def clear(self) -> None:
        """Clear all tracking data."""
        self._failures.clear()
