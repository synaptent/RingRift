"""Voter Health Monitor with Auto-Demotion.

January 2026: Created as part of P2P critical hardening (Phase 1).
Problem: Unhealthy voters remain in quorum, causing:
- Slow elections (waiting for timeout on unresponsive voters)
- Reduced quorum availability (dead voters still counted)
- Split-brain risk (inconsistent voter state)

Solution: Monitor voter health and auto-demote unhealthy voters:
1. Track response time, error rate, consensus participation
2. Demote voters that exceed thresholds
3. Allow re-promotion after recovery (with cooloff period)

Usage:
    from scripts.p2p.voter_health_monitor import (
        VoterHealthMonitor,
        get_voter_health_monitor,
    )

    # Get singleton monitor
    monitor = get_voter_health_monitor()

    # Start background monitoring loop
    await monitor.start()

    # Check if voter should be demoted
    health = monitor.get_voter_health(voter_id)
    if health.should_demote:
        await monitor.demote_voter(voter_id)

    # Check if voter can be re-promoted
    if monitor.can_repromote(voter_id):
        await monitor.repromote_voter(voter_id)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import TYPE_CHECKING, Any, Callable, Awaitable

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class VoterHealthState(Enum):
    """Health states for a voter node."""

    HEALTHY = "healthy"  # Responsive, low error rate
    DEGRADED = "degraded"  # Slow or some errors
    UNHEALTHY = "unhealthy"  # Many errors, slow responses
    DEMOTED = "demoted"  # Removed from active quorum
    RECOVERING = "recovering"  # In cooloff before re-promotion


@dataclass
class VoterHealthConfig:
    """Configuration for voter health monitoring."""

    # Response time thresholds (milliseconds)
    response_time_degraded_ms: float = 5000.0  # 5s = degraded
    response_time_unhealthy_ms: float = 10000.0  # 10s = unhealthy (tightened from 15s)

    # Error rate thresholds (fraction of requests)
    error_rate_degraded: float = 0.10  # 10% errors = degraded
    error_rate_unhealthy: float = 0.20  # 20% errors = unhealthy (tightened from 30%)

    # Consecutive failures before demotion
    consecutive_failures_demote: int = 4  # (tightened from 5)

    # Time window for calculating metrics (seconds)
    metrics_window_seconds: float = 300.0  # 5 minutes

    # Cooloff before re-promotion (seconds)
    repromote_cooloff_seconds: float = 600.0  # 10 minutes

    # Health check interval (seconds)
    check_interval_seconds: float = 30.0

    # Minimum voters to maintain (never demote below this)
    min_voters_for_quorum: int = 3

    # Require this many healthy checks before re-promotion
    repromote_healthy_checks: int = 5


@dataclass
class VoterMetrics:
    """Health metrics for a single voter."""

    voter_id: str
    state: VoterHealthState = VoterHealthState.HEALTHY

    # Rolling window metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time_ms: float = 0.0

    # Recent history
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    last_check_time: float = 0.0

    # Demotion tracking
    demoted_at: float = 0.0
    demotion_reason: str = ""
    healthy_checks_since_demotion: int = 0

    def record_success(self, response_time_ms: float) -> None:
        """Record a successful health check."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_response_time_ms += response_time_ms
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        self.last_check_time = time.time()

        if self.state == VoterHealthState.DEMOTED:
            self.healthy_checks_since_demotion += 1

    def record_failure(self, reason: str = "") -> None:
        """Record a failed health check."""
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = time.time()
        self.last_check_time = time.time()

        if self.state == VoterHealthState.DEMOTED:
            self.healthy_checks_since_demotion = 0

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def avg_response_time_ms(self) -> float:
        """Calculate average response time."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time_ms / self.successful_requests


@dataclass
class VoterHealthStatus:
    """Current health status of a voter."""

    voter_id: str
    state: VoterHealthState
    should_demote: bool = False
    can_repromote: bool = False
    error_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    consecutive_failures: int = 0
    demoted_at: float | None = None
    demotion_reason: str = ""
    time_since_demotion: float | None = None
    healthy_checks_since_demotion: int = 0


class VoterHealthMonitor:
    """Monitor voter health and manage auto-demotion.

    January 2026: Tracks voter node health and automatically demotes
    voters that become unhealthy. Also manages re-promotion after recovery.

    Thread Safety:
        Uses RLock for thread-safe metric updates and state access.
    """

    _instance: VoterHealthMonitor | None = None
    _lock = RLock()

    def __init__(
        self,
        config: VoterHealthConfig | None = None,
        get_voters_fn: Callable[[], list[str]] | None = None,
        check_voter_fn: Callable[[str], Awaitable[tuple[bool, float]]] | None = None,
        demote_callback: Callable[[str], Awaitable[bool]] | None = None,
        repromote_callback: Callable[[str], Awaitable[bool]] | None = None,
    ):
        """Initialize voter health monitor.

        Args:
            config: Health monitoring configuration
            get_voters_fn: Function to get current voter list
            check_voter_fn: Async function to check voter health
                           Returns: (success, response_time_ms)
            demote_callback: Called when voter is demoted
            repromote_callback: Called when voter is re-promoted
        """
        self.config = config or VoterHealthConfig()
        self._get_voters_fn = get_voters_fn
        self._check_voter_fn = check_voter_fn
        self._demote_callback = demote_callback
        self._repromote_callback = repromote_callback

        self._metrics: dict[str, VoterMetrics] = {}
        self._demoted_voters: set[str] = set()
        self._running = False
        self._task: asyncio.Task | None = None
        self._lock = RLock()

    @classmethod
    def get_instance(
        cls,
        config: VoterHealthConfig | None = None,
    ) -> VoterHealthMonitor:
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config=config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            if cls._instance and cls._instance._running:
                cls._instance._running = False
            cls._instance = None

    def configure(
        self,
        get_voters_fn: Callable[[], list[str]] | None = None,
        check_voter_fn: Callable[[str], Awaitable[tuple[bool, float]]] | None = None,
        demote_callback: Callable[[str], Awaitable[bool]] | None = None,
        repromote_callback: Callable[[str], Awaitable[bool]] | None = None,
    ) -> None:
        """Configure callbacks after initialization."""
        if get_voters_fn:
            self._get_voters_fn = get_voters_fn
        if check_voter_fn:
            self._check_voter_fn = check_voter_fn
        if demote_callback:
            self._demote_callback = demote_callback
        if repromote_callback:
            self._repromote_callback = repromote_callback

    async def start(self) -> None:
        """Start the health monitoring loop."""
        if self._running:
            return

        self._running = True
        self._task = safe_create_task(self._monitor_loop(), name="voter-health-monitor")
        logger.info("[VoterHealth] Started health monitoring")

    async def stop(self) -> None:
        """Stop the health monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[VoterHealth] Stopped health monitoring")

    async def _monitor_loop(self) -> None:
        """Background loop for health monitoring."""
        while self._running:
            try:
                await self._check_all_voters()
                await asyncio.sleep(self.config.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[VoterHealth] Error in monitor loop: {e}")
                await asyncio.sleep(self.config.check_interval_seconds)

    async def _check_all_voters(self) -> None:
        """Check health of all voters in parallel.

        Jan 7, 2026: Converted from sequential to parallel checks to prevent
        cascading failures where one slow voter blocks all subsequent checks.
        Previously: 10 voters × 15s timeout = 150s worst case
        Now: 10 voters checked in ~15s worst case (parallel)
        """
        if not self._get_voters_fn or not self._check_voter_fn:
            return

        voters = self._get_voters_fn()
        if not voters:
            return

        # Check all voters in parallel
        async def check_single_voter(voter_id: str) -> tuple[str, bool, float]:
            """Check a single voter and return results."""
            try:
                success, response_time_ms = await self._check_voter_fn(voter_id)
                return (voter_id, success, response_time_ms)
            except Exception as e:
                logger.debug(f"[VoterHealth] Error checking {voter_id}: {e}")
                return (voter_id, False, 0.0)

        # Run all checks in parallel
        results = await asyncio.gather(
            *[check_single_voter(voter_id) for voter_id in voters],
            return_exceptions=True,
        )

        # Process results sequentially (record + demotion decisions)
        for result in results:
            if isinstance(result, Exception):
                logger.debug(f"[VoterHealth] Check failed with exception: {result}")
                continue

            voter_id, success, response_time_ms = result
            self._record_check(voter_id, success, response_time_ms)

            # Check for demotion
            status = self.get_voter_health(voter_id)
            if status.should_demote:
                await self.demote_voter(voter_id)
            elif status.can_repromote:
                await self.repromote_voter(voter_id)

    def _record_check(
        self,
        voter_id: str,
        success: bool,
        response_time_ms: float,
    ) -> None:
        """Record health check result."""
        with self._lock:
            if voter_id not in self._metrics:
                self._metrics[voter_id] = VoterMetrics(voter_id=voter_id)

            metrics = self._metrics[voter_id]

            if success:
                metrics.record_success(response_time_ms)
                # Update state based on response time
                if response_time_ms > self.config.response_time_unhealthy_ms:
                    metrics.state = VoterHealthState.UNHEALTHY
                elif response_time_ms > self.config.response_time_degraded_ms:
                    metrics.state = VoterHealthState.DEGRADED
                elif metrics.state != VoterHealthState.DEMOTED:
                    metrics.state = VoterHealthState.HEALTHY
            else:
                metrics.record_failure()
                if metrics.consecutive_failures >= self.config.consecutive_failures_demote:
                    if metrics.state != VoterHealthState.DEMOTED:
                        metrics.state = VoterHealthState.UNHEALTHY

    def get_voter_health(self, voter_id: str) -> VoterHealthStatus:
        """Get current health status of a voter."""
        with self._lock:
            metrics = self._metrics.get(voter_id)
            if not metrics:
                return VoterHealthStatus(
                    voter_id=voter_id,
                    state=VoterHealthState.HEALTHY,
                )

            # Check if should demote
            should_demote = (
                metrics.state != VoterHealthState.DEMOTED
                and (
                    metrics.consecutive_failures >= self.config.consecutive_failures_demote
                    or metrics.error_rate >= self.config.error_rate_unhealthy
                )
            )

            # Check if can re-promote
            can_repromote = False
            if metrics.state == VoterHealthState.DEMOTED and metrics.demoted_at:
                time_since_demotion = time.time() - metrics.demoted_at
                if (
                    time_since_demotion >= self.config.repromote_cooloff_seconds
                    and metrics.healthy_checks_since_demotion >= self.config.repromote_healthy_checks
                ):
                    can_repromote = True

            return VoterHealthStatus(
                voter_id=voter_id,
                state=metrics.state,
                should_demote=should_demote,
                can_repromote=can_repromote,
                error_rate=metrics.error_rate,
                avg_response_time_ms=metrics.avg_response_time_ms,
                consecutive_failures=metrics.consecutive_failures,
                demoted_at=metrics.demoted_at if metrics.demoted_at else None,
                demotion_reason=metrics.demotion_reason,
                time_since_demotion=(
                    time.time() - metrics.demoted_at if metrics.demoted_at else None
                ),
                healthy_checks_since_demotion=metrics.healthy_checks_since_demotion,
            )

    async def demote_voter(self, voter_id: str) -> bool:
        """Demote an unhealthy voter.

        Args:
            voter_id: Voter to demote

        Returns:
            True if demotion succeeded, False otherwise
        """
        # Check if we have enough voters
        if self._get_voters_fn:
            active_voters = [
                v for v in self._get_voters_fn()
                if v not in self._demoted_voters
            ]
            if len(active_voters) <= self.config.min_voters_for_quorum:
                logger.warning(
                    f"[VoterHealth] Cannot demote {voter_id}: would go below quorum"
                )
                return False

        with self._lock:
            if voter_id not in self._metrics:
                return False

            metrics = self._metrics[voter_id]
            if metrics.state == VoterHealthState.DEMOTED:
                return True  # Already demoted

            reason = f"failures={metrics.consecutive_failures}, error_rate={metrics.error_rate:.2%}"
            metrics.state = VoterHealthState.DEMOTED
            metrics.demoted_at = time.time()
            metrics.demotion_reason = reason
            metrics.healthy_checks_since_demotion = 0
            self._demoted_voters.add(voter_id)

        logger.warning(f"[VoterHealth] Demoted voter {voter_id}: {reason}")

        # Call demotion callback
        if self._demote_callback:
            try:
                await self._demote_callback(voter_id)
            except Exception as e:
                logger.error(f"[VoterHealth] Demotion callback failed: {e}")

        # Emit event
        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import emit_event

            emit_event(DataEventType.VOTER_DEMOTED, {
                "voter_id": voter_id,
                "reason": reason,
                "active_voters": len(self._get_voters_fn()) - len(self._demoted_voters) if self._get_voters_fn else 0,
            })
        except ImportError:
            pass

        return True

    async def repromote_voter(self, voter_id: str) -> bool:
        """Re-promote a recovered voter.

        Args:
            voter_id: Voter to re-promote

        Returns:
            True if re-promotion succeeded, False otherwise
        """
        with self._lock:
            metrics = self._metrics.get(voter_id)
            if not metrics or metrics.state != VoterHealthState.DEMOTED:
                return False

            # Check eligibility
            if metrics.demoted_at:
                time_since_demotion = time.time() - metrics.demoted_at
                if time_since_demotion < self.config.repromote_cooloff_seconds:
                    return False
                if metrics.healthy_checks_since_demotion < self.config.repromote_healthy_checks:
                    return False

            metrics.state = VoterHealthState.RECOVERING
            metrics.demoted_at = 0.0
            metrics.demotion_reason = ""
            self._demoted_voters.discard(voter_id)

        logger.info(f"[VoterHealth] Re-promoted voter {voter_id}")

        # Call re-promotion callback
        if self._repromote_callback:
            try:
                await self._repromote_callback(voter_id)
            except Exception as e:
                logger.error(f"[VoterHealth] Re-promotion callback failed: {e}")

        # Update state to healthy after callback
        with self._lock:
            if voter_id in self._metrics:
                self._metrics[voter_id].state = VoterHealthState.HEALTHY

        # Emit event
        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import emit_event

            emit_event(DataEventType.VOTER_PROMOTED, {
                "voter_id": voter_id,
                "active_voters": len(self._get_voters_fn()) if self._get_voters_fn else 0,
            })
        except ImportError:
            pass

        return True

    def get_all_health(self) -> dict[str, VoterHealthStatus]:
        """Get health status for all monitored voters."""
        with self._lock:
            return {
                voter_id: self.get_voter_health(voter_id)
                for voter_id in self._metrics
            }

    def get_demoted_voters(self) -> list[str]:
        """Get list of demoted voters."""
        with self._lock:
            return list(self._demoted_voters)

    def get_summary(self) -> dict[str, Any]:
        """Get health monitoring summary."""
        with self._lock:
            all_health = self.get_all_health()
            states = [h.state for h in all_health.values()]

            return {
                "total_voters": len(all_health),
                "healthy": sum(1 for s in states if s == VoterHealthState.HEALTHY),
                "degraded": sum(1 for s in states if s == VoterHealthState.DEGRADED),
                "unhealthy": sum(1 for s in states if s == VoterHealthState.UNHEALTHY),
                "demoted": sum(1 for s in states if s == VoterHealthState.DEMOTED),
                "recovering": sum(1 for s in states if s == VoterHealthState.RECOVERING),
                "demoted_voters": list(self._demoted_voters),
                "running": self._running,
            }


# =============================================================================
# Module-level accessors
# =============================================================================


def get_voter_health_monitor() -> VoterHealthMonitor:
    """Get the voter health monitor singleton."""
    return VoterHealthMonitor.get_instance()


__all__ = [
    "VoterHealthMonitor",
    "VoterHealthConfig",
    "VoterHealthState",
    "VoterHealthStatus",
    "VoterMetrics",
    "get_voter_health_monitor",
]
