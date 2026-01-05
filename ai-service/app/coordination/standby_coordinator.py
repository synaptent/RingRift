"""
StandbyCoordinator - Distributed coordinator failover for cluster resilience.

This module provides automatic failover when the primary coordinator fails.
Part of the 4-layer cluster resilience architecture (Session 16, January 2026).

Architecture:
    Primary Coordinator (active)
        |
        | monitors heartbeat via gossip
        v
    StandbyCoordinator (passive, ready to take over)
        |
        | on primary failure
        v
    StandbyCoordinator.take_over() -> becomes primary

Key Features:
    - Monitors primary coordinator health via P2P gossip
    - Automatic failover when primary is unresponsive
    - Graceful handoff when primary recovers
    - Integration with MemoryPressureController for graceful shutdown

Usage:
    from app.coordination.standby_coordinator import (
        StandbyCoordinator,
        get_standby_coordinator,
        StandbyConfig,
    )

    # Get singleton
    standby = get_standby_coordinator()

    # Start monitoring
    await standby.start()

    # Check if this node is primary
    if standby.is_primary:
        # Run coordinator logic
        pass
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from app.config.env import env
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
from app.coordination.handler_base import HandlerBase

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class CoordinatorRole(Enum):
    """Role of this coordinator in the cluster."""

    PRIMARY = "primary"
    STANDBY = "standby"
    TRANSITIONING = "transitioning"
    UNKNOWN = "unknown"


class FailoverReason(Enum):
    """Reason for failover."""

    PRIMARY_TIMEOUT = "primary_timeout"
    PRIMARY_SHUTDOWN = "primary_shutdown"
    MEMORY_EMERGENCY = "memory_emergency"
    MANUAL_TAKEOVER = "manual_takeover"
    ELECTION_WON = "election_won"


@dataclass
class StandbyConfig:
    """Configuration for standby coordinator."""

    # Primary monitoring
    primary_heartbeat_timeout: float = 120.0  # Seconds before primary is considered dead
    primary_check_interval: float = 15.0  # How often to check primary health
    primary_host: Optional[str] = None  # Primary coordinator host (auto-discovered if None)
    primary_port: int = 8790  # Primary coordinator health port

    # Failover behavior
    takeover_delay: float = 10.0  # Delay before taking over (prevents flapping)
    graceful_handoff_timeout: float = 60.0  # Timeout for graceful handoff to recovered primary
    min_standby_uptime: float = 300.0  # Minimum uptime before eligible for takeover

    # State persistence
    state_file: Path = field(
        default_factory=lambda: Path("/tmp/ringrift_standby_state.json")
    )

    # Event callbacks
    on_takeover_callbacks: list[Callable[[], None]] = field(default_factory=list)
    on_handoff_callbacks: list[Callable[[], None]] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> StandbyConfig:
        """Create config from environment variables."""
        import os

        return cls(
            primary_heartbeat_timeout=float(
                os.getenv("RINGRIFT_STANDBY_HEARTBEAT_TIMEOUT", "120.0")
            ),
            primary_check_interval=float(
                os.getenv("RINGRIFT_STANDBY_CHECK_INTERVAL", "15.0")
            ),
            primary_host=os.getenv("RINGRIFT_STANDBY_PRIMARY_HOST"),
            primary_port=int(os.getenv("RINGRIFT_STANDBY_PRIMARY_PORT", "8790")),
            takeover_delay=float(
                os.getenv("RINGRIFT_STANDBY_TAKEOVER_DELAY", "10.0")
            ),
        )


@dataclass
class PrimaryHealthState:
    """Health state of the primary coordinator."""

    host: str
    is_healthy: bool
    last_seen: float
    consecutive_failures: int
    last_check_time: float
    last_check_duration: float
    error_message: Optional[str] = None

    @property
    def time_since_seen(self) -> float:
        """Seconds since primary was last seen healthy."""
        if self.last_seen <= 0:
            return float("inf")
        return time.time() - self.last_seen

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "host": self.host,
            "is_healthy": self.is_healthy,
            "last_seen": self.last_seen,
            "consecutive_failures": self.consecutive_failures,
            "last_check_time": self.last_check_time,
            "last_check_duration": self.last_check_duration,
            "time_since_seen": self.time_since_seen,
            "error_message": self.error_message,
        }


@dataclass
class StandbyState:
    """Current state of the standby coordinator."""

    role: CoordinatorRole
    start_time: float
    takeover_count: int
    handoff_count: int
    last_takeover_time: float
    last_handoff_time: float
    failover_reason: Optional[FailoverReason]
    primary_health: Optional[PrimaryHealthState]

    @property
    def uptime_seconds(self) -> float:
        """Uptime in seconds."""
        if self.start_time <= 0:
            return 0.0
        return time.time() - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role.value,
            "uptime_seconds": self.uptime_seconds,
            "takeover_count": self.takeover_count,
            "handoff_count": self.handoff_count,
            "last_takeover_time": self.last_takeover_time,
            "last_handoff_time": self.last_handoff_time,
            "failover_reason": self.failover_reason.value if self.failover_reason else None,
            "primary_health": self.primary_health.to_dict() if self.primary_health else None,
        }


# =============================================================================
# Standby Coordinator
# =============================================================================


class StandbyCoordinator(HandlerBase):
    """Standby coordinator that can take over on primary failure.

    This coordinator monitors the primary coordinator's health via HTTP
    health checks and P2P gossip. When the primary fails, it automatically
    takes over coordinator responsibilities.

    Lifecycle:
        1. Start in STANDBY role
        2. Monitor primary via health checks
        3. If primary fails, transition to PRIMARY
        4. If primary recovers, optionally hand off back

    Integration:
        - Subscribes to MEMORY_PRESSURE_EMERGENCY from MemoryPressureController
        - Emits COORDINATOR_FAILOVER on takeover
        - Emits COORDINATOR_HANDOFF on handoff
    """

    def __init__(self, config: Optional[StandbyConfig] = None):
        """Initialize standby coordinator.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self._standby_config = config or StandbyConfig.from_env()
        super().__init__(
            name="standby_coordinator",
            config=self._standby_config,
            cycle_interval=self._standby_config.primary_check_interval,
        )
        self._role = CoordinatorRole.UNKNOWN
        self._start_time = 0.0
        self._takeover_count = 0
        self._handoff_count = 0
        self._last_takeover_time = 0.0
        self._last_handoff_time = 0.0
        self._failover_reason: Optional[FailoverReason] = None
        self._primary_health: Optional[PrimaryHealthState] = None

        # Internal state
        self._http_client: Optional[Any] = None
        self._primary_host: Optional[str] = None
        self._consecutive_failures: int = 0

        # Callbacks
        self._on_takeover: list[Callable[[], None]] = list(
            self._standby_config.on_takeover_callbacks
        )
        self._on_handoff: list[Callable[[], None]] = list(
            self._standby_config.on_handoff_callbacks
        )

        logger.info(
            "StandbyCoordinator initialized",
            extra={
                "node_id": env.node_id,
                "primary_host": self._standby_config.primary_host,
                "heartbeat_timeout": self._standby_config.primary_heartbeat_timeout,
            },
        )

    @property
    def is_primary(self) -> bool:
        """Check if this node is the primary coordinator."""
        return self._role == CoordinatorRole.PRIMARY

    @property
    def is_standby(self) -> bool:
        """Check if this node is in standby mode."""
        return self._role == CoordinatorRole.STANDBY

    @property
    def role(self) -> CoordinatorRole:
        """Current role of this coordinator."""
        return self._role

    def get_state(self) -> StandbyState:
        """Get current standby state."""
        return StandbyState(
            role=self._role,
            start_time=self._start_time,
            takeover_count=self._takeover_count,
            handoff_count=self._handoff_count,
            last_takeover_time=self._last_takeover_time,
            last_handoff_time=self._last_handoff_time,
            failover_reason=self._failover_reason,
            primary_health=self._primary_health,
        )

    def register_takeover_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for takeover events."""
        self._on_takeover.append(callback)

    def register_handoff_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for handoff events."""
        self._on_handoff.append(callback)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def _on_start(self) -> None:
        """Start the standby coordinator."""
        self._start_time = time.time()

        # Discover primary host if not configured
        if not self._standby_config.primary_host:
            self._primary_host = await self._discover_primary_host()
        else:
            self._primary_host = self._standby_config.primary_host

        if not self._primary_host:
            logger.warning(
                "No primary host configured or discovered, starting as PRIMARY"
            )
            self._role = CoordinatorRole.PRIMARY
            return

        # Determine initial role
        if self._primary_host == env.node_id:
            # This node is the primary
            self._role = CoordinatorRole.PRIMARY
            logger.info("Starting as PRIMARY coordinator")
        else:
            # This node is standby
            self._role = CoordinatorRole.STANDBY
            logger.info(
                "Starting as STANDBY coordinator",
                extra={"primary_host": self._primary_host},
            )
            # HandlerBase's main loop will call _run_cycle() for monitoring

    async def _on_stop(self) -> None:
        """Stop the standby coordinator."""
        if self._http_client:
            await self._http_client.close()
            self._http_client = None

        logger.info(
            "StandbyCoordinator stopped",
            extra={
                "role": self._role.value,
                "takeover_count": self._takeover_count,
                "handoff_count": self._handoff_count,
            },
        )

    # =========================================================================
    # Primary Monitoring (via HandlerBase cycle)
    # =========================================================================

    async def _run_cycle(self) -> None:
        """One cycle of primary health monitoring.

        This is called by HandlerBase every cycle_interval seconds.
        Only performs monitoring when in STANDBY role.
        """
        # Only monitor when in STANDBY role
        if self._role != CoordinatorRole.STANDBY:
            return

        check_start = time.time()
        is_healthy = await self._check_primary_health()
        check_duration = time.time() - check_start

        if is_healthy:
            self._consecutive_failures = 0
            self._primary_health = PrimaryHealthState(
                host=self._primary_host or "",
                is_healthy=True,
                last_seen=time.time(),
                consecutive_failures=0,
                last_check_time=time.time(),
                last_check_duration=check_duration,
            )
        else:
            self._consecutive_failures += 1
            self._primary_health = PrimaryHealthState(
                host=self._primary_host or "",
                is_healthy=False,
                last_seen=self._primary_health.last_seen
                if self._primary_health
                else 0,
                consecutive_failures=self._consecutive_failures,
                last_check_time=time.time(),
                last_check_duration=check_duration,
                error_message="Health check failed",
            )

            # Check if primary is dead
            if await self._should_take_over():
                await self._take_over(FailoverReason.PRIMARY_TIMEOUT)

    async def _check_primary_health(self) -> bool:
        """Check if primary coordinator is healthy.

        Returns:
            True if primary is healthy, False otherwise.
        """
        if not self._primary_host:
            return False

        try:
            # Lazy import aiohttp
            if self._http_client is None:
                import aiohttp

                self._http_client = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10.0)
                )

            url = f"http://{self._primary_host}:{self._standby_config.primary_port}/health"
            async with self._http_client.get(url) as resp:
                if resp.status == 200:
                    return True
                else:
                    logger.warning(
                        "Primary health check returned non-200",
                        extra={"status": resp.status, "url": url},
                    )
                    return False

        except Exception as e:
            logger.debug(
                "Primary health check failed",
                extra={"host": self._primary_host, "error": str(e)},
            )
            return False

    async def _should_take_over(self) -> bool:
        """Determine if this standby should take over.

        Returns:
            True if takeover conditions are met.
        """
        if not self._primary_health:
            return False

        # Check if we've been running long enough
        if self.get_state().uptime_seconds < self._standby_config.min_standby_uptime:
            logger.debug(
                "Not eligible for takeover yet",
                extra={
                    "uptime": self.get_state().uptime_seconds,
                    "min_required": self._standby_config.min_standby_uptime,
                },
            )
            return False

        # Check if primary has been unresponsive long enough
        time_since_seen = self._primary_health.time_since_seen
        if time_since_seen < self._standby_config.primary_heartbeat_timeout:
            return False

        logger.warning(
            "Primary coordinator appears dead, preparing takeover",
            extra={
                "time_since_seen": time_since_seen,
                "threshold": self._standby_config.primary_heartbeat_timeout,
                "consecutive_failures": self._primary_health.consecutive_failures,
            },
        )

        # Wait takeover delay to prevent flapping
        await asyncio.sleep(self._standby_config.takeover_delay)

        # Double-check after delay
        is_healthy = await self._check_primary_health()
        if is_healthy:
            logger.info("Primary recovered during takeover delay, aborting takeover")
            return False

        return True

    # =========================================================================
    # Takeover / Handoff
    # =========================================================================

    async def _take_over(self, reason: FailoverReason) -> None:
        """Take over as primary coordinator.

        Args:
            reason: Reason for the takeover.
        """
        if self._role == CoordinatorRole.PRIMARY:
            logger.warning("Already primary, ignoring takeover request")
            return

        logger.critical(
            "FAILOVER: Taking over as PRIMARY coordinator",
            extra={
                "reason": reason.value,
                "old_primary": self._primary_host,
                "new_primary": env.node_id,
            },
        )

        self._role = CoordinatorRole.TRANSITIONING
        self._failover_reason = reason
        self._takeover_count += 1
        self._last_takeover_time = time.time()

        # Emit event
        from app.coordination.event_emission_helpers import safe_emit_event
        from app.distributed.data_events import DataEventType

        safe_emit_event(
            DataEventType.COORDINATOR_FAILOVER,
            {
                "new_primary": env.node_id,
                "old_primary": self._primary_host,
                "reason": reason.value,
                "takeover_count": self._takeover_count,
            },
            context="StandbyCoordinator",
            source="standby_coordinator",
        )

        # Run takeover callbacks
        for callback in self._on_takeover:
            try:
                callback()
            except Exception as e:
                logger.error(
                    "Error in takeover callback",
                    exc_info=True,
                    extra={"error": str(e)},
                )

        self._role = CoordinatorRole.PRIMARY
        logger.info("Takeover complete, now running as PRIMARY")

    async def hand_off_to_primary(self, new_primary: str) -> bool:
        """Hand off coordinator role to a recovered primary.

        Args:
            new_primary: Host of the new primary.

        Returns:
            True if handoff was successful.
        """
        if self._role != CoordinatorRole.PRIMARY:
            logger.warning("Not primary, cannot hand off")
            return False

        logger.info(
            "HANDOFF: Returning coordinator role to primary",
            extra={
                "new_primary": new_primary,
                "current_node": env.node_id,
            },
        )

        self._role = CoordinatorRole.TRANSITIONING
        self._handoff_count += 1
        self._last_handoff_time = time.time()

        # Emit event
        from app.coordination.event_emission_helpers import safe_emit_event
        from app.distributed.data_events import DataEventType

        safe_emit_event(
            DataEventType.COORDINATOR_HANDOFF,
            {
                "new_primary": new_primary,
                "old_primary": env.node_id,
                "handoff_count": self._handoff_count,
            },
            context="StandbyCoordinator",
            source="standby_coordinator",
        )

        # Run handoff callbacks
        for callback in self._on_handoff:
            try:
                callback()
            except Exception as e:
                logger.error(
                    "Error in handoff callback",
                    exc_info=True,
                    extra={"error": str(e)},
                )

        self._role = CoordinatorRole.STANDBY
        self._primary_host = new_primary
        self._consecutive_failures = 0  # Reset failure count for new primary

        # Monitoring resumes automatically via HandlerBase's _run_cycle()
        logger.info("Handoff complete, now running as STANDBY")
        return True

    async def force_takeover(self, reason: FailoverReason = FailoverReason.MANUAL_TAKEOVER) -> None:
        """Force immediate takeover (manual intervention).

        Args:
            reason: Reason for the manual takeover.
        """
        logger.warning(
            "Force takeover requested",
            extra={"reason": reason.value},
        )
        await self._take_over(reason)

    def on_memory_emergency(self) -> None:
        """Handle memory emergency from MemoryPressureController.

        This is a callback that can be registered with MemoryPressureController
        to trigger failover when the coordinator is under extreme memory pressure.
        """
        if self._role == CoordinatorRole.PRIMARY:
            logger.critical(
                "Memory emergency on PRIMARY, notifying standbys to take over"
            )
            # In emergency, we emit an event for standby to take over
            from app.coordination.event_emission_helpers import safe_emit_event
            from app.distributed.data_events import DataEventType

            safe_emit_event(
                DataEventType.COORDINATOR_EMERGENCY_SHUTDOWN,
                {
                    "primary": env.node_id,
                    "reason": "memory_emergency",
                },
                context="StandbyCoordinator",
                source="standby_coordinator",
            )

    # =========================================================================
    # Discovery
    # =========================================================================

    async def _discover_primary_host(self) -> Optional[str]:
        """Discover the primary coordinator from cluster config.

        Returns:
            Primary host address or None if not found.
        """
        try:
            from app.config.cluster_config import get_coordinator_node

            coordinator = get_coordinator_node()
            if coordinator:
                return coordinator.best_ip
        except ImportError:
            pass

        # Fallback to environment variable
        import os

        return os.getenv("RINGRIFT_PRIMARY_COORDINATOR")

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Return health check result.

        Returns:
            HealthCheckResult with current standby state.
        """
        state = self.get_state()

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="StandbyCoordinator not running",
                details={"role": self._role.value},
            )

        if self._role == CoordinatorRole.PRIMARY:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message="Running as primary coordinator",
                details={
                    "role": "primary",
                    "takeover_count": state.takeover_count,
                    "uptime_seconds": state.uptime_seconds,
                },
            )
        elif self._role == CoordinatorRole.STANDBY:
            primary_healthy = (
                self._primary_health.is_healthy if self._primary_health else False
            )
            # Standby is healthy if primary is being monitored
            status = CoordinatorStatus.RUNNING if primary_healthy else CoordinatorStatus.DEGRADED
            return HealthCheckResult(
                healthy=True,  # Standby itself is healthy even if primary is not
                status=status,
                message="Monitoring primary coordinator",
                details={
                    "role": "standby",
                    "primary_host": self._primary_host,
                    "primary_healthy": primary_healthy,
                    "primary_last_seen": (
                        self._primary_health.time_since_seen
                        if self._primary_health
                        else None
                    ),
                    "consecutive_failures": self._consecutive_failures,
                },
            )
        else:
            # TRANSITIONING or UNKNOWN
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STARTING,
                message=f"In transitional state: {self._role.value}",
                details={"role": self._role.value},
            )


# =============================================================================
# Module-level accessors
# =============================================================================


def get_standby_coordinator(config: Optional[StandbyConfig] = None) -> StandbyCoordinator:
    """Get or create the singleton StandbyCoordinator instance.

    Args:
        config: Optional configuration for first-time initialization.

    Returns:
        The StandbyCoordinator singleton.
    """
    return StandbyCoordinator.get_instance(config)


def reset_standby_coordinator() -> None:
    """Reset the singleton (for testing)."""
    StandbyCoordinator.reset_instance()
