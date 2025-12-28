"""Sync stall detection and automatic failover handler.

This module provides automatic detection and recovery from stalled sync operations.
When a sync stalls or times out, it:
1. Marks the stalled source as temporarily unavailable
2. Tracks stall history with penalties
3. Provides alternative healthy sources for retry
4. Records recovery statistics

This is CRITICAL for cluster reliability in multi-node environments.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class SyncStallHandler:
    """Handles automatic failover from stalled sync sources to healthy mirrors.

    Features:
    - Stall detection with configurable timeout
    - Per-host penalty tracking with automatic expiry
    - Alternative source selection excluding stalled hosts
    - Recovery tracking and statistics
    - Bounded retry attempts to prevent infinite loops

    Example:
        from app.config.coordination_defaults import SyncDefaults

        handler = SyncStallHandler(stall_penalty_seconds=300, max_retries=3)

        # Check if sync is stalled (use centralized timeout)
        if handler.check_stall(sync_id, started_at, timeout=SyncDefaults.STALL_DETECTION_TIMEOUT):
            # Record the stall
            handler.record_stall(host="node-5", sync_id=sync_id)

            # Get alternative source
            alt = handler.get_alternative_source(exclude=["node-5"])
            if alt:
                handler.record_recovery(sync_id, alt)
    """

    def __init__(
        self,
        stall_penalty_seconds: float = 300.0,
        max_retries: int = 3,
    ):
        """Initialize the stall handler.

        Args:
            stall_penalty_seconds: Time to deprioritize stalled sources (default: 5 min)
            max_retries: Maximum retry attempts before giving up (default: 3)
        """
        self.stall_penalty_seconds = stall_penalty_seconds
        self.max_retries = max_retries

        # Per-host penalty tracking: host -> penalty_until_timestamp
        self._stalled_sources: dict[str, float] = {}

        # Statistics
        self._stall_count = 0
        self._recovery_count = 0
        self._failed_recoveries = 0

        logger.info(
            f"[SyncStallHandler] Initialized with penalty={stall_penalty_seconds}s, "
            f"max_retries={max_retries}"
        )

    def check_stall(self, sync_id: str, started_at: float, timeout: float) -> bool:
        """Check if a sync operation has stalled.

        Args:
            sync_id: Unique identifier for the sync operation
            started_at: Timestamp when sync started
            timeout: Timeout in seconds

        Returns:
            True if sync has stalled (exceeded timeout)
        """
        elapsed = time.time() - started_at
        if elapsed > timeout:
            logger.warning(
                f"[SyncStallHandler] Sync {sync_id} stalled: {elapsed:.1f}s > {timeout}s"
            )
            return True
        return False

    def record_stall(self, host: str, sync_id: str) -> None:
        """Record a stall event and apply penalty to the source host.

        Args:
            host: The host that experienced the stall
            sync_id: Unique identifier for the sync operation
        """
        if not host:
            logger.warning(f"[SyncStallHandler] Cannot record stall for empty host (sync={sync_id})")
            return

        self._stall_count += 1
        penalty_until = time.time() + self.stall_penalty_seconds

        self._stalled_sources[host] = penalty_until

        logger.warning(
            f"[SyncStallHandler] Stall #{self._stall_count}: host={host} sync={sync_id} "
            f"penalized for {self.stall_penalty_seconds}s"
        )

    def get_alternative_source(
        self,
        exclude: list[str] | None = None,
        all_sources: list[str] | None = None,
    ) -> str | None:
        """Get a healthy alternative source, excluding stalled/excluded hosts.

        Args:
            exclude: Additional hosts to exclude from selection
            all_sources: Pool of available sources (if None, returns None when all excluded)

        Returns:
            A healthy source host, or None if no alternatives available
        """
        exclude = exclude or []
        current_time = time.time()

        # Build excluded set: manual exclusions + currently penalized sources
        excluded_set = set(exclude)
        for source, penalty_until in self._stalled_sources.items():
            if penalty_until > current_time:
                excluded_set.add(source)

        # If no source pool provided, we can't suggest alternatives
        if all_sources is None:
            if excluded_set:
                logger.debug(
                    f"[SyncStallHandler] {len(excluded_set)} sources excluded, "
                    "but no source pool provided"
                )
            return None

        # Find available sources
        available = [s for s in all_sources if s not in excluded_set]

        if not available:
            logger.warning(
                f"[SyncStallHandler] No alternative sources available "
                f"(pool={len(all_sources)}, excluded={len(excluded_set)})"
            )
            return None

        # Return first available (could be randomized or load-balanced in future)
        selected = available[0]
        logger.info(
            f"[SyncStallHandler] Selected alternative source: {selected} "
            f"(from {len(available)} available)"
        )
        return selected

    def record_recovery(self, sync_id: str, new_source: str) -> None:
        """Record a successful failover to a new source.

        Args:
            sync_id: Unique identifier for the sync operation
            new_source: The new source host used for recovery
        """
        self._recovery_count += 1
        logger.info(
            f"[SyncStallHandler] Recovery #{self._recovery_count}: sync={sync_id} "
            f"failed over to {new_source}"
        )

    def record_failed_recovery(self, sync_id: str, reason: str) -> None:
        """Record a failed recovery attempt.

        Args:
            sync_id: Unique identifier for the sync operation
            reason: Reason for recovery failure
        """
        self._failed_recoveries += 1
        logger.error(
            f"[SyncStallHandler] Failed recovery #{self._failed_recoveries}: "
            f"sync={sync_id} reason={reason}"
        )

    def is_source_available(self, source: str) -> bool:
        """Check if a source is currently available (not penalized).

        Args:
            source: Source identifier

        Returns:
            True if source is available (not under penalty)
        """
        penalty_until = self._stalled_sources.get(source, 0.0)
        return time.time() >= penalty_until

    def clear_penalties(self) -> int:
        """Clear all expired penalties.

        Returns:
            Number of penalties cleared
        """
        current_time = time.time()
        expired = [
            source
            for source, penalty_until in self._stalled_sources.items()
            if penalty_until <= current_time
        ]

        for source in expired:
            del self._stalled_sources[source]

        if expired:
            logger.debug(f"[SyncStallHandler] Cleared {len(expired)} expired penalties")

        return len(expired)

    def get_stats(self) -> dict[str, Any]:
        """Get handler statistics.

        Returns:
            Dict with stall counts, active penalties, and recovery stats
        """
        current_time = time.time()
        active_penalties = {
            source: penalty_until - current_time
            for source, penalty_until in self._stalled_sources.items()
            if penalty_until > current_time
        }

        return {
            "stall_count": self._stall_count,
            "recovery_count": self._recovery_count,
            "failed_recoveries": self._failed_recoveries,
            "active_penalties": len(active_penalties),
            "penalized_sources": list(active_penalties.keys()),
            "penalty_details": {
                source: f"{remaining:.1f}s"
                for source, remaining in active_penalties.items()
            },
            "config": {
                "stall_penalty_seconds": self.stall_penalty_seconds,
                "max_retries": self.max_retries,
            },
        }

    def reset(self) -> None:
        """Reset all handler state (for testing)."""
        self._stalled_sources.clear()
        self._stall_count = 0
        self._recovery_count = 0
        self._failed_recoveries = 0

    def health_check(self) -> "HealthCheckResult":
        """Check stall handler health for daemon monitoring.

        December 2025: Added for unified daemon health monitoring.

        Returns:
            HealthCheckResult with health status.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        try:
            stats = self.get_stats()
            active_penalties = stats.get("active_penalties", 0)
            stall_count = stats.get("stall_count", 0)
            recovery_count = stats.get("recovery_count", 0)
            failed_recoveries = stats.get("failed_recoveries", 0)

            # Calculate recovery success rate
            total_stalls = stall_count
            recovery_rate = (
                recovery_count / total_stalls * 100
                if total_stalls > 0
                else 100.0
            )

            # Degraded if too many failed recoveries
            if failed_recoveries > 5:
                return HealthCheckResult(
                    healthy=True,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"SyncStallHandler degraded: {failed_recoveries} failed recoveries",
                    details={
                        "stall_count": stall_count,
                        "recovery_count": recovery_count,
                        "failed_recoveries": failed_recoveries,
                        "active_penalties": active_penalties,
                        "recovery_rate_percent": recovery_rate,
                    },
                )

            # Healthy
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"SyncStallHandler healthy: {stall_count} stalls, {recovery_count} recoveries, {active_penalties} active penalties",
                details={
                    "stall_count": stall_count,
                    "recovery_count": recovery_count,
                    "failed_recoveries": failed_recoveries,
                    "active_penalties": active_penalties,
                    "recovery_rate_percent": recovery_rate,
                    "penalized_sources": stats.get("penalized_sources", []),
                },
            )

        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"SyncStallHandler health check error: {e}",
            )


# Module-level singleton
_stall_handler: SyncStallHandler | None = None


def get_stall_handler(
    stall_penalty_seconds: float = 300.0,
    max_retries: int = 3,
) -> SyncStallHandler:
    """Get or create the module-level SyncStallHandler singleton.

    Args:
        stall_penalty_seconds: Penalty duration for stalled sources
        max_retries: Maximum retry attempts

    Returns:
        SyncStallHandler instance
    """
    global _stall_handler
    if _stall_handler is None:
        _stall_handler = SyncStallHandler(
            stall_penalty_seconds=stall_penalty_seconds,
            max_retries=max_retries,
        )
    return _stall_handler


def reset_stall_handler() -> None:
    """Reset the module-level singleton (for testing)."""
    global _stall_handler
    if _stall_handler is not None:
        _stall_handler.reset()
    _stall_handler = None
