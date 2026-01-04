"""
ClusterResilienceOrchestrator - Unified health aggregation and early warning.

This module provides a single source of truth for cluster health,
aggregating signals from all resilience layers. Part of the 4-layer
cluster resilience architecture (Session 16, January 2026).

Layers Aggregated:
    Layer 1: Hierarchical Process Supervision (Sentinel → Watchdog)
    Layer 2: Proactive Memory Management (MemoryPressureController)
    Layer 3: Distributed Coordinator Resilience (StandbyCoordinator)
    Layer 4: This orchestrator (unified health aggregation)

Key Features:
    - Computes overall resilience score (0.0-1.0)
    - Emits early warning events before cascading failures
    - Tracks component health with weighted scoring
    - Recommends recovery actions based on degradation patterns

Usage:
    from app.coordination.cluster_resilience_orchestrator import (
        ClusterResilienceOrchestrator,
        get_resilience_orchestrator,
        ResilienceScore,
    )

    # Get singleton
    orchestrator = get_resilience_orchestrator()

    # Get current resilience score
    score = orchestrator.get_resilience_score()
    print(f"Cluster resilience: {score.overall:.1%}")

    # Start continuous monitoring
    await orchestrator.start()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from app.config.env import env
from app.coordination.coordinator_base import CoordinatorBase
from app.coordination.singleton_mixin import SingletonMixin

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class ResilienceLevel(Enum):
    """Overall cluster resilience level."""

    HEALTHY = "healthy"  # All components healthy
    WARNING = "warning"  # Some degradation, monitor closely
    DEGRADED = "degraded"  # Significant issues, action needed
    CRITICAL = "critical"  # Imminent failure, immediate action required


class RecoveryAction(Enum):
    """Recommended recovery actions."""

    NONE = "none"
    PAUSE_SELFPLAY = "pause_selfplay"  # Reduce load
    PAUSE_TRAINING = "pause_training"  # Reduce memory pressure
    RESTART_DAEMONS = "restart_daemons"  # Restart unhealthy daemons
    RESTART_P2P = "restart_p2p"  # Restart P2P orchestrator
    TRIGGER_ELECTION = "trigger_election"  # Force leader election
    TRIGGER_FAILOVER = "trigger_failover"  # Trigger coordinator failover
    GRACEFUL_SHUTDOWN = "graceful_shutdown"  # Graceful cluster shutdown


@dataclass
class ResilienceConfig:
    """Configuration for resilience orchestrator."""

    # Monitoring
    check_interval: float = 30.0  # Seconds between checks
    early_warning_threshold: float = 0.70  # Score below this emits warning

    # Scoring weights
    memory_weight: float = 0.30  # Memory pressure
    coordinator_weight: float = 0.30  # Coordinator health
    quorum_weight: float = 0.25  # P2P quorum
    daemon_weight: float = 0.15  # Daemon health

    # Level thresholds (score < threshold = level)
    critical_threshold: float = 0.45
    degraded_threshold: float = 0.65
    warning_threshold: float = 0.85

    @classmethod
    def from_env(cls) -> ResilienceConfig:
        """Create config from environment variables."""
        import os

        return cls(
            check_interval=float(
                os.getenv("RINGRIFT_RESILIENCE_CHECK_INTERVAL", "30.0")
            ),
            early_warning_threshold=float(
                os.getenv("RINGRIFT_RESILIENCE_WARNING_THRESHOLD", "0.70")
            ),
        )


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    score: float  # 0.0-1.0
    is_healthy: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    last_check_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "score": self.score,
            "is_healthy": self.is_healthy,
            "message": self.message,
            "details": self.details,
            "last_check_time": self.last_check_time,
        }


@dataclass
class ResilienceScore:
    """Overall resilience score with component breakdown."""

    overall: float  # 0.0-1.0 weighted score
    level: ResilienceLevel
    components: dict[str, ComponentHealth]
    recommended_actions: list[RecoveryAction]
    timestamp: float
    degraded_components: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall": self.overall,
            "level": self.level.value,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "recommended_actions": [a.value for a in self.recommended_actions],
            "timestamp": self.timestamp,
            "degraded_components": self.degraded_components,
        }


# =============================================================================
# Cluster Resilience Orchestrator
# =============================================================================


class ClusterResilienceOrchestrator(CoordinatorBase, SingletonMixin):
    """Unified cluster health aggregation and early warning.

    This orchestrator provides a single source of truth for cluster health
    by aggregating signals from all resilience layers:

    - Layer 1: Process supervision (heartbeat freshness)
    - Layer 2: Memory pressure (tier-based)
    - Layer 3: Coordinator failover (primary/standby status)
    - Layer 4: P2P quorum and daemon health

    The resilience score is computed as a weighted average of component
    health scores, with configurable weights (default: memory 30%,
    coordinator 30%, quorum 25%, daemon 15%).
    """

    def __init__(self, config: Optional[ResilienceConfig] = None):
        """Initialize the orchestrator.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        CoordinatorBase.__init__(self)
        self._config = config or ResilienceConfig.from_env()
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_score: Optional[ResilienceScore] = None
        self._last_warning_time = 0.0
        self._warning_cooldown = 300.0  # 5 minutes between warnings

        # Callbacks for resilience events
        self._on_degraded: list[Callable[[ResilienceScore], None]] = []
        self._on_critical: list[Callable[[ResilienceScore], None]] = []

        logger.info(
            "ClusterResilienceOrchestrator initialized",
            extra={
                "check_interval": self._config.check_interval,
                "warning_threshold": self._config.early_warning_threshold,
            },
        )

    @property
    def name(self) -> str:
        """Return coordinator name."""
        return "cluster_resilience_orchestrator"

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def _on_start(self) -> None:
        """Start the orchestrator."""
        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(),
            name="resilience_monitor",
        )
        logger.info("ClusterResilienceOrchestrator started")

    async def _on_stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("ClusterResilienceOrchestrator stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                score = self.get_resilience_score()
                self._last_score = score

                # Check for early warning
                if score.overall < self._config.early_warning_threshold:
                    await self._emit_early_warning(score)

                # Check for level transitions
                if score.level == ResilienceLevel.DEGRADED:
                    for callback in self._on_degraded:
                        try:
                            callback(score)
                        except Exception as e:
                            logger.error(f"Degraded callback error: {e}")

                elif score.level == ResilienceLevel.CRITICAL:
                    for callback in self._on_critical:
                        try:
                            callback(score)
                        except Exception as e:
                            logger.error(f"Critical callback error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resilience monitor loop: {e}", exc_info=True)

            await asyncio.sleep(self._config.check_interval)

    # =========================================================================
    # Health Score Computation
    # =========================================================================

    def get_resilience_score(self) -> ResilienceScore:
        """Compute the current cluster resilience score.

        Returns:
            ResilienceScore with overall score, level, and component breakdown.
        """
        now = time.time()
        components: dict[str, ComponentHealth] = {}

        # Collect component health
        components["memory"] = self._get_memory_health()
        components["coordinator"] = self._get_coordinator_health()
        components["quorum"] = self._get_quorum_health()
        components["daemon"] = self._get_daemon_health()

        # Compute weighted overall score
        weights = {
            "memory": self._config.memory_weight,
            "coordinator": self._config.coordinator_weight,
            "quorum": self._config.quorum_weight,
            "daemon": self._config.daemon_weight,
        }

        overall = sum(
            components[k].score * weights[k]
            for k in weights
            if k in components
        )

        # Determine level
        if overall < self._config.critical_threshold:
            level = ResilienceLevel.CRITICAL
        elif overall < self._config.degraded_threshold:
            level = ResilienceLevel.DEGRADED
        elif overall < self._config.warning_threshold:
            level = ResilienceLevel.WARNING
        else:
            level = ResilienceLevel.HEALTHY

        # Get degraded components
        degraded = [
            name for name, health in components.items()
            if not health.is_healthy
        ]

        # Recommend actions based on level and degraded components
        actions = self._recommend_actions(level, degraded, components)

        return ResilienceScore(
            overall=overall,
            level=level,
            components=components,
            recommended_actions=actions,
            timestamp=now,
            degraded_components=degraded,
        )

    def _get_memory_health(self) -> ComponentHealth:
        """Get memory pressure health score."""
        try:
            from app.coordination.memory_pressure_controller import (
                MemoryPressureTier,
                get_memory_pressure_controller,
            )

            controller = get_memory_pressure_controller()
            state = controller.get_state()

            # Map tier to score
            tier_scores = {
                MemoryPressureTier.NORMAL: 1.0,
                MemoryPressureTier.CAUTION: 0.75,
                MemoryPressureTier.WARNING: 0.50,
                MemoryPressureTier.CRITICAL: 0.25,
                MemoryPressureTier.EMERGENCY: 0.0,
            }

            score = tier_scores.get(state.current_tier, 0.5)
            is_healthy = state.current_tier in (
                MemoryPressureTier.NORMAL,
                MemoryPressureTier.CAUTION,
            )

            return ComponentHealth(
                name="memory",
                score=score,
                is_healthy=is_healthy,
                message=f"Memory at {state.memory_percent:.1f}% ({state.current_tier.value})",
                details={
                    "tier": state.current_tier.value,
                    "percent": state.memory_percent,
                    "tier_changes": state.tier_changes,
                },
                last_check_time=time.time(),
            )

        except ImportError:
            # Fallback to psutil
            try:
                import psutil

                mem = psutil.virtual_memory()
                percent = mem.percent
                score = max(0.0, 1.0 - (percent / 100.0))
                is_healthy = percent < 80

                return ComponentHealth(
                    name="memory",
                    score=score,
                    is_healthy=is_healthy,
                    message=f"Memory at {percent:.1f}%",
                    details={"percent": percent},
                    last_check_time=time.time(),
                )
            except ImportError:
                return ComponentHealth(
                    name="memory",
                    score=0.5,
                    is_healthy=True,
                    message="Memory status unknown (psutil not available)",
                    last_check_time=time.time(),
                )

    def _get_coordinator_health(self) -> ComponentHealth:
        """Get coordinator resilience health score."""
        try:
            from app.coordination.standby_coordinator import get_standby_coordinator

            standby = get_standby_coordinator()
            state = standby.get_state()

            if standby.is_primary:
                return ComponentHealth(
                    name="coordinator",
                    score=1.0,
                    is_healthy=True,
                    message="Running as primary coordinator",
                    details={"role": "primary", "takeover_count": state.takeover_count},
                    last_check_time=time.time(),
                )
            else:
                # Standby - check primary health
                if state.primary_health and state.primary_health.is_healthy:
                    return ComponentHealth(
                        name="coordinator",
                        score=0.9,  # Slight penalty for being standby
                        is_healthy=True,
                        message="Standby monitoring healthy primary",
                        details={
                            "role": "standby",
                            "primary_host": state.primary_health.host,
                        },
                        last_check_time=time.time(),
                    )
                else:
                    # Primary not responding
                    return ComponentHealth(
                        name="coordinator",
                        score=0.3,
                        is_healthy=False,
                        message="Primary coordinator not responding",
                        details={
                            "role": "standby",
                            "primary_health": state.primary_health.to_dict()
                            if state.primary_health
                            else None,
                        },
                        last_check_time=time.time(),
                    )

        except ImportError:
            return ComponentHealth(
                name="coordinator",
                score=0.5,
                is_healthy=True,
                message="StandbyCoordinator not available",
                last_check_time=time.time(),
            )

    def _get_quorum_health(self) -> ComponentHealth:
        """Get P2P quorum health score."""
        try:
            # Try to get quorum health from P2P health coordinator
            from scripts.p2p.health_coordinator import get_health_coordinator

            hc = get_health_coordinator()
            health = hc.get_health()

            quorum_score = health.quorum_health
            is_healthy = quorum_score >= 0.5

            return ComponentHealth(
                name="quorum",
                score=quorum_score,
                is_healthy=is_healthy,
                message=f"Quorum health: {quorum_score:.1%}",
                details={
                    "overall_health": health.overall_health,
                    "level": health.level.value if hasattr(health.level, "value") else str(health.level),
                },
                last_check_time=time.time(),
            )

        except ImportError:
            # Fallback - try to check P2P status directly
            try:
                import json
                import urllib.request

                with urllib.request.urlopen(
                    "http://localhost:8770/status", timeout=5
                ) as resp:
                    data = json.loads(resp.read())
                    alive_peers = data.get("alive_peers", 0)
                    # Simple heuristic: 5+ peers = healthy
                    score = min(1.0, alive_peers / 5.0)
                    is_healthy = alive_peers >= 3

                    return ComponentHealth(
                        name="quorum",
                        score=score,
                        is_healthy=is_healthy,
                        message=f"{alive_peers} alive peers",
                        details={"alive_peers": alive_peers},
                        last_check_time=time.time(),
                    )
            except Exception:
                return ComponentHealth(
                    name="quorum",
                    score=0.5,
                    is_healthy=True,
                    message="P2P status unknown",
                    last_check_time=time.time(),
                )

    def _get_daemon_health(self) -> ComponentHealth:
        """Get daemon manager health score."""
        try:
            from app.coordination.daemon_manager import get_daemon_manager

            dm = get_daemon_manager()
            health = dm.get_all_daemon_health()

            total = len(health)
            if total == 0:
                return ComponentHealth(
                    name="daemon",
                    score=1.0,
                    is_healthy=True,
                    message="No daemons running",
                    last_check_time=time.time(),
                )

            healthy_count = sum(
                1 for h in health.values()
                if h.get("status") in ("healthy", "running")
            )
            score = healthy_count / total
            is_healthy = score >= 0.8

            unhealthy = [
                name for name, h in health.items()
                if h.get("status") not in ("healthy", "running")
            ]

            return ComponentHealth(
                name="daemon",
                score=score,
                is_healthy=is_healthy,
                message=f"{healthy_count}/{total} daemons healthy",
                details={
                    "total": total,
                    "healthy": healthy_count,
                    "unhealthy": unhealthy[:5],  # First 5
                },
                last_check_time=time.time(),
            )

        except ImportError:
            return ComponentHealth(
                name="daemon",
                score=0.5,
                is_healthy=True,
                message="DaemonManager not available",
                last_check_time=time.time(),
            )

    def _recommend_actions(
        self,
        level: ResilienceLevel,
        degraded: list[str],
        components: dict[str, ComponentHealth],
    ) -> list[RecoveryAction]:
        """Recommend recovery actions based on health state."""
        actions: list[RecoveryAction] = []

        if level == ResilienceLevel.HEALTHY:
            return [RecoveryAction.NONE]

        # Memory issues
        if "memory" in degraded:
            mem_score = components["memory"].score
            if mem_score < 0.25:
                actions.append(RecoveryAction.GRACEFUL_SHUTDOWN)
            elif mem_score < 0.5:
                actions.extend([
                    RecoveryAction.PAUSE_TRAINING,
                    RecoveryAction.PAUSE_SELFPLAY,
                ])

        # Coordinator issues
        if "coordinator" in degraded:
            actions.append(RecoveryAction.TRIGGER_FAILOVER)

        # Quorum issues
        if "quorum" in degraded:
            quorum_score = components["quorum"].score
            if quorum_score < 0.3:
                actions.append(RecoveryAction.TRIGGER_ELECTION)
            else:
                actions.append(RecoveryAction.RESTART_P2P)

        # Daemon issues
        if "daemon" in degraded:
            actions.append(RecoveryAction.RESTART_DAEMONS)

        return actions if actions else [RecoveryAction.NONE]

    # =========================================================================
    # Event Emission
    # =========================================================================

    async def _emit_early_warning(self, score: ResilienceScore) -> None:
        """Emit early warning event if not on cooldown."""
        now = time.time()
        if now - self._last_warning_time < self._warning_cooldown:
            return

        self._last_warning_time = now

        logger.warning(
            "CLUSTER RESILIENCE DEGRADED",
            extra={
                "score": score.overall,
                "level": score.level.value,
                "degraded": score.degraded_components,
                "actions": [a.value for a in score.recommended_actions],
            },
        )

        try:
            from app.coordination.event_router import emit_event
            from app.distributed.data_events import DataEventType

            emit_event(
                DataEventType.CLUSTER_HEALTH_CHANGED,
                {
                    "score": score.overall,
                    "level": score.level.value,
                    "degraded_components": score.degraded_components,
                    "recommended_actions": [a.value for a in score.recommended_actions],
                    "node_id": env.node_id,
                },
            )
        except ImportError:
            pass

    # =========================================================================
    # Registration
    # =========================================================================

    def register_degraded_callback(
        self, callback: Callable[[ResilienceScore], None]
    ) -> None:
        """Register callback for degraded state."""
        self._on_degraded.append(callback)

    def register_critical_callback(
        self, callback: Callable[[ResilienceScore], None]
    ) -> None:
        """Register callback for critical state."""
        self._on_critical.append(callback)

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> Any:
        """Return health check result."""
        from app.coordination.health_check_result import HealthCheckResult

        score = self._last_score or self.get_resilience_score()

        if score.level == ResilienceLevel.CRITICAL:
            return HealthCheckResult.unhealthy(
                f"Cluster resilience CRITICAL: {score.overall:.1%}"
            )
        elif score.level == ResilienceLevel.DEGRADED:
            return HealthCheckResult.degraded(
                f"Cluster resilience degraded: {score.overall:.1%}"
            )

        return HealthCheckResult(
            healthy=True,
            status=score.level.value,
            message=f"Cluster resilience: {score.overall:.1%}",
            details={"score": score.to_dict()},
        )


# =============================================================================
# Module-level accessors
# =============================================================================

_orchestrator_instance: Optional[ClusterResilienceOrchestrator] = None


def get_resilience_orchestrator(
    config: Optional[ResilienceConfig] = None,
) -> ClusterResilienceOrchestrator:
    """Get or create the singleton ClusterResilienceOrchestrator instance.

    Args:
        config: Optional configuration for first-time initialization.

    Returns:
        The ClusterResilienceOrchestrator singleton.
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = ClusterResilienceOrchestrator(config)
    return _orchestrator_instance


def reset_resilience_orchestrator() -> None:
    """Reset the singleton (for testing)."""
    global _orchestrator_instance
    _orchestrator_instance = None
