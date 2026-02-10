"""Base class for P2P sub-orchestrators.

January 2026: Created as part of P2POrchestrator decomposition.
Each sub-orchestrator handles a specific domain (leadership, network, sync, jobs)
and is composed into the main P2POrchestrator.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""

    healthy: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "healthy": self.healthy,
            "message": self.message,
            "details": self.details,
        }


def get_job_attr(job: Any, attr: str, default: Any = None) -> Any:
    """Get attribute from a job entry that may be a ClusterJob object or a plain dict.

    Feb 2026: Some code paths store jobs as ClusterJob objects, others as plain dicts.
    This helper handles both transparently.
    """
    if isinstance(job, dict):
        return job.get(attr, default)
    return getattr(job, attr, default)


def set_job_attr(job: Any, attr: str, value: Any) -> None:
    """Set attribute on a job entry that may be a ClusterJob object or a plain dict."""
    if isinstance(job, dict):
        job[attr] = value
    else:
        setattr(job, attr, value)


class BaseOrchestrator(ABC):
    """Base class for P2P sub-orchestrators.

    Each sub-orchestrator:
    - Has access to the parent P2POrchestrator for shared state
    - Manages a specific domain of functionality
    - Provides a health_check() method for monitoring
    - Has its own logger for clear log attribution

    Usage:
        class LeadershipOrchestrator(BaseOrchestrator):
            @property
            def name(self) -> str:
                return "leadership"

            def health_check(self) -> HealthCheckResult:
                return HealthCheckResult(
                    healthy=True,
                    message="Leadership orchestrator healthy",
                    details={"is_leader": self._p2p.is_leader()},
                )
    """

    def __init__(self, p2p: "P2POrchestrator") -> None:
        """Initialize the orchestrator.

        Args:
            p2p: The parent P2POrchestrator instance for accessing shared state.
        """
        self._p2p = p2p
        self._logger = logging.getLogger(f"p2p.orchestrator.{self.name}")
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this orchestrator (e.g., 'leadership', 'network')."""
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> HealthCheckResult:
        """Check the health of this orchestrator.

        Returns:
            HealthCheckResult with healthy status and diagnostic details.
        """
        raise NotImplementedError

    def initialize(self) -> None:
        """Initialize the orchestrator after P2POrchestrator is fully constructed.

        Override this method to perform any initialization that depends on
        other orchestrators or managers being available.
        """
        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        """Return whether this orchestrator has been initialized."""
        return self._initialized

    # Convenience properties for common P2POrchestrator attributes

    @property
    def node_id(self) -> str:
        """Return the node ID of the parent orchestrator."""
        return self._p2p.node_id

    @property
    def is_leader(self) -> bool:
        """Return whether this node is currently the leader."""
        return self._p2p.is_leader()

    @property
    def leader_id(self) -> str | None:
        """Return the current leader ID, or None if no leader."""
        return getattr(self._p2p, "leader_id", None)

    def _log_debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message."""
        self._logger.debug(message, *args, **kwargs)

    def _log_info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an info message."""
        self._logger.info(message, *args, **kwargs)

    def _log_warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message."""
        self._logger.warning(message, *args, **kwargs)

    def _log_error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message."""
        self._logger.error(message, *args, **kwargs)

    def _safe_emit_event(self, event_type: str, data: dict[str, Any]) -> bool:
        """Safely emit an event via the P2POrchestrator's event system.

        Args:
            event_type: The type of event to emit.
            data: Event data dictionary.

        Returns:
            True if event was emitted, False if event system unavailable.
        """
        try:
            if hasattr(self._p2p, "safe_emit_event"):
                self._p2p.safe_emit_event(event_type, data)
                return True
            elif hasattr(self._p2p, "_safe_emit_event"):
                self._p2p._safe_emit_event(event_type, data)
                return True
        except Exception as e:
            self._log_debug(f"Failed to emit event {event_type}: {e}")
        return False
