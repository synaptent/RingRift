"""SystemHealthMonitorDaemon - Global system health monitoring with pipeline pause (December 2025).

This module provides cluster-wide health aggregation and automatic pipeline pause
when critical failures occur. It wraps UnifiedHealthManager and adds:

1. Aggregate health score (0-100) across all nodes
2. Pipeline pause when health drops below threshold
3. Automatic resume when health recovers
4. System-wide critical event emission

Health Score Components:
- Node availability (40%): % of expected nodes online
- Circuit breaker status (25%): % of circuits closed
- Error rate (20%): Recent error rate vs threshold
- Recovery success rate (15%): Recent recovery success %

Pipeline Pause Triggers (any one triggers pause):
- Health score < 40
- >50% nodes offline
- Critical circuits broken (training, evaluation, promotion)
- >10 unrecovered errors in 5 minutes

Usage:
    from app.coordination.system_health_monitor import (
        SystemHealthMonitorDaemon,
        get_system_health,
        is_pipeline_paused,
    )

    # Get singleton
    monitor = get_system_health()

    # Check health
    score = monitor.get_health_score()
    if score < 50:
        print("System unhealthy!")

    # Check if pipeline is paused
    if is_pipeline_paused():
        print("Pipeline paused due to system health")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================


class SystemHealthLevel(Enum):
    """System health levels."""

    HEALTHY = "healthy"  # 80-100
    DEGRADED = "degraded"  # 60-79
    UNHEALTHY = "unhealthy"  # 40-59
    CRITICAL = "critical"  # 0-39


class PipelineState(Enum):
    """Pipeline operational state."""

    RUNNING = "running"
    PAUSED = "paused"
    RECOVERING = "recovering"


@dataclass
class SystemHealthScore:
    """Aggregate system health score."""

    score: int  # 0-100
    level: SystemHealthLevel
    components: dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # Component scores (0-100 each)
    node_availability: float = 100.0
    circuit_health: float = 100.0
    error_rate: float = 100.0  # Inverted: 100 = no errors
    recovery_success: float = 100.0

    # Pause triggers
    pause_triggers: list[str] = field(default_factory=list)


@dataclass
class SystemHealthConfig:
    """Configuration for system health monitoring."""

    # Check interval
    check_interval_seconds: int = 30

    # Health score thresholds
    healthy_threshold: int = 80
    degraded_threshold: int = 60
    unhealthy_threshold: int = 40

    # Pause triggers
    pause_health_threshold: int = 40
    pause_node_offline_percent: float = 0.5  # 50%
    pause_error_burst_count: int = 10
    pause_error_burst_window: int = 300  # 5 minutes

    # Critical circuits that trigger immediate pause if broken
    critical_circuits: list[str] = field(
        default_factory=lambda: ["training", "evaluation", "promotion"]
    )

    # Resume thresholds (hysteresis)
    resume_health_threshold: int = 60
    resume_delay_seconds: int = 120  # Wait 2 min before resuming

    # Expected nodes (0 = auto-discover)
    expected_nodes: int = 0

    # Component weights for score calculation
    node_weight: float = 0.40
    circuit_weight: float = 0.25
    error_weight: float = 0.20
    recovery_weight: float = 0.15


# =============================================================================
# SystemHealthMonitorDaemon
# =============================================================================


class SystemHealthMonitorDaemon:
    """Global system health monitor with pipeline pause capability.

    This daemon periodically checks overall system health and automatically
    pauses the pipeline when critical thresholds are breached.
    """

    def __init__(self, config: SystemHealthConfig | None = None):
        """Initialize the system health monitor.

        Args:
            config: Health monitoring configuration
        """
        self.config = config or SystemHealthConfig()

        # State
        self._running = False
        self._pipeline_state = PipelineState.RUNNING
        self._paused_at: float = 0.0
        self._pause_reason: str = ""
        self._last_check_time: float = 0.0
        self._last_health_score: SystemHealthScore | None = None

        # History for trending
        self._health_history: list[SystemHealthScore] = []
        self._max_history = 120  # 1 hour at 30s intervals

        # Error tracking for burst detection
        self._recent_errors: list[float] = []

        # Callbacks
        self._pause_callbacks: list[callable] = []
        self._resume_callbacks: list[callable] = []
        self._health_callbacks: list[callable] = []

        # Background task
        self._monitor_task: asyncio.Task | None = None

    # =========================================================================
    # Health Score Calculation
    # =========================================================================

    def _get_health_manager(self):
        """Get the UnifiedHealthManager singleton."""
        try:
            from app.coordination.unified_health_manager import get_health_manager

            return get_health_manager()
        except ImportError:
            return None

    def _get_cluster_status(self) -> dict[str, Any]:
        """Get cluster status from P2P or cluster monitor."""
        try:
            # Try P2P status first
            import requests

            response = requests.get("http://localhost:8770/status", timeout=5)
            if response.ok:
                return response.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            logger.debug(f"P2P status request failed: {e}")
        except Exception as e:
            logger.debug(f"P2P status request error: {e}")

        return {}

    def _calculate_node_availability(self) -> float:
        """Calculate node availability score (0-100)."""
        health_manager = self._get_health_manager()
        cluster_status = self._get_cluster_status()

        if not health_manager:
            return 100.0  # Assume healthy if no data

        # Get online/offline from health manager
        nodes_tracked = len(health_manager._node_states)
        nodes_online = sum(
            1 for s in health_manager._node_states.values() if s.is_online
        )
        nodes_offline = nodes_tracked - nodes_online

        # Also check cluster status
        alive_peers = cluster_status.get("alive_peers", 0)

        # Determine expected nodes
        expected = self.config.expected_nodes
        if expected == 0:
            # Auto-discover from cluster or node states
            expected = max(nodes_tracked, alive_peers, 1)

        # Calculate availability
        actual_online = max(nodes_online, alive_peers)
        availability = (actual_online / expected) * 100 if expected > 0 else 100.0

        return min(100.0, availability)

    def _calculate_circuit_health(self) -> float:
        """Calculate circuit breaker health score (0-100)."""
        health_manager = self._get_health_manager()
        if not health_manager:
            return 100.0

        total_circuits = len(health_manager._circuit_breakers)
        if total_circuits == 0:
            return 100.0

        from app.distributed.circuit_breaker import CircuitState

        open_circuits = sum(
            1
            for cb in health_manager._circuit_breakers.values()
            if cb.state == CircuitState.OPEN
        )

        # Circuits closed percentage
        closed_percent = ((total_circuits - open_circuits) / total_circuits) * 100

        # Extra penalty for critical circuits
        critical_open = [
            c
            for c in self.config.critical_circuits
            if c in health_manager._circuit_breakers
            and health_manager._circuit_breakers[c].state == CircuitState.OPEN
        ]

        if critical_open:
            # Heavy penalty for critical circuits
            penalty = len(critical_open) * 20
            closed_percent = max(0, closed_percent - penalty)

        return closed_percent

    def _calculate_error_rate_score(self) -> float:
        """Calculate error rate score (0-100, inverted: higher = fewer errors)."""
        health_manager = self._get_health_manager()
        if not health_manager:
            return 100.0

        # Check recent errors (last 5 minutes)
        now = time.time()
        window = self.config.pause_error_burst_window
        recent_errors = [
            e for e in health_manager._errors if now - e.timestamp < window
        ]

        error_count = len(recent_errors)

        # Score based on error count
        # 0 errors = 100, threshold errors = 0
        threshold = self.config.pause_error_burst_count
        if error_count >= threshold:
            return 0.0

        score = ((threshold - error_count) / threshold) * 100
        return max(0.0, min(100.0, score))

    def _calculate_recovery_success(self) -> float:
        """Calculate recovery success rate (0-100)."""
        health_manager = self._get_health_manager()
        if not health_manager:
            return 100.0

        total = health_manager._total_recoveries
        successful = health_manager._successful_recoveries

        if total == 0:
            return 100.0  # No recoveries needed = healthy

        success_rate = (successful / total) * 100
        return success_rate

    def calculate_health_score(self) -> SystemHealthScore:
        """Calculate aggregate system health score."""
        # Calculate component scores
        node_availability = self._calculate_node_availability()
        circuit_health = self._calculate_circuit_health()
        error_rate = self._calculate_error_rate_score()
        recovery_success = self._calculate_recovery_success()

        # Weighted aggregate
        score = (
            node_availability * self.config.node_weight
            + circuit_health * self.config.circuit_weight
            + error_rate * self.config.error_weight
            + recovery_success * self.config.recovery_weight
        )

        score = int(max(0, min(100, score)))

        # Determine level
        if score >= self.config.healthy_threshold:
            level = SystemHealthLevel.HEALTHY
        elif score >= self.config.degraded_threshold:
            level = SystemHealthLevel.DEGRADED
        elif score >= self.config.unhealthy_threshold:
            level = SystemHealthLevel.UNHEALTHY
        else:
            level = SystemHealthLevel.CRITICAL

        # Check pause triggers
        pause_triggers = self._check_pause_triggers(
            score, node_availability, circuit_health, error_rate
        )

        return SystemHealthScore(
            score=score,
            level=level,
            components={
                "node_availability": round(node_availability, 1),
                "circuit_health": round(circuit_health, 1),
                "error_rate": round(error_rate, 1),
                "recovery_success": round(recovery_success, 1),
            },
            node_availability=node_availability,
            circuit_health=circuit_health,
            error_rate=error_rate,
            recovery_success=recovery_success,
            pause_triggers=pause_triggers,
        )

    def _check_pause_triggers(
        self,
        score: int,
        node_availability: float,
        circuit_health: float,
        error_rate: float,
    ) -> list[str]:
        """Check for conditions that should trigger pipeline pause."""
        triggers = []

        # Health score threshold
        if score < self.config.pause_health_threshold:
            triggers.append(f"health_score_critical:{score}")

        # Node offline threshold
        offline_percent = (100 - node_availability) / 100
        if offline_percent >= self.config.pause_node_offline_percent:
            triggers.append(f"nodes_offline:{offline_percent:.0%}")

        # Critical circuit broken
        health_manager = self._get_health_manager()
        if health_manager:
            from app.distributed.circuit_breaker import CircuitState

            for circuit_name in self.config.critical_circuits:
                if circuit_name in health_manager._circuit_breakers:
                    cb = health_manager._circuit_breakers[circuit_name]
                    if cb.state == CircuitState.OPEN:
                        triggers.append(f"critical_circuit_open:{circuit_name}")

        # Error burst
        if error_rate == 0:
            triggers.append("error_burst_detected")

        return triggers

    # =========================================================================
    # Pipeline Pause/Resume
    # =========================================================================

    def _should_pause(self, health: SystemHealthScore) -> bool:
        """Determine if pipeline should be paused."""
        if self._pipeline_state == PipelineState.PAUSED:
            return False  # Already paused

        return len(health.pause_triggers) > 0

    def _should_resume(self, health: SystemHealthScore) -> bool:
        """Determine if pipeline should resume."""
        if self._pipeline_state != PipelineState.PAUSED:
            return False  # Not paused

        # Check if enough time has passed
        if time.time() - self._paused_at < self.config.resume_delay_seconds:
            return False

        # Check if health recovered above resume threshold
        if health.score < self.config.resume_health_threshold:
            return False

        # Check if all pause triggers cleared
        return len(health.pause_triggers) == 0

    async def _pause_pipeline(self, health: SystemHealthScore) -> None:
        """Pause the training pipeline."""
        if self._pipeline_state == PipelineState.PAUSED:
            return

        self._pipeline_state = PipelineState.PAUSED
        self._paused_at = time.time()
        self._pause_reason = ", ".join(health.pause_triggers)

        logger.warning(
            f"[SystemHealthMonitor] PIPELINE PAUSED - Score: {health.score}, "
            f"Triggers: {self._pause_reason}"
        )

        # Emit SYSTEM_CRITICAL event
        await self._emit_system_critical(health)

        # Notify callbacks
        for callback in self._pause_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(health)
                else:
                    callback(health)
            except Exception as e:
                logger.error(f"[SystemHealthMonitor] Pause callback failed: {e}")

    async def _resume_pipeline(self, health: SystemHealthScore) -> None:
        """Resume the training pipeline."""
        if self._pipeline_state != PipelineState.PAUSED:
            return

        self._pipeline_state = PipelineState.RECOVERING
        pause_duration = time.time() - self._paused_at

        logger.info(
            f"[SystemHealthMonitor] PIPELINE RESUMING - Score: {health.score}, "
            f"Paused for: {pause_duration:.0f}s"
        )

        # Emit recovery event
        await self._emit_system_recovered(health, pause_duration)

        # Notify callbacks
        for callback in self._resume_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(health)
                else:
                    callback(health)
            except Exception as e:
                logger.error(f"[SystemHealthMonitor] Resume callback failed: {e}")

        self._pipeline_state = PipelineState.RUNNING
        self._paused_at = 0.0
        self._pause_reason = ""

    async def _emit_system_critical(self, health: SystemHealthScore) -> None:
        """Emit SYSTEM_CRITICAL event."""
        try:
            from app.coordination.event_router import get_router
            from app.distributed.data_events import DataEventType

            router = get_router()
            await router.publish(
                "system_critical",
                {
                    "health_score": health.score,
                    "level": health.level.value,
                    "triggers": health.pause_triggers,
                    "components": health.components,
                    "timestamp": health.timestamp,
                },
            )
        except Exception as e:
            logger.debug(f"[SystemHealthMonitor] Failed to emit system_critical: {e}")

    async def _emit_system_recovered(
        self, health: SystemHealthScore, pause_duration: float
    ) -> None:
        """Emit SYSTEM_RECOVERED event."""
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            await router.publish(
                "system_recovered",
                {
                    "health_score": health.score,
                    "level": health.level.value,
                    "pause_duration_seconds": pause_duration,
                    "components": health.components,
                    "timestamp": health.timestamp,
                },
            )
        except Exception as e:
            logger.debug(f"[SystemHealthMonitor] Failed to emit system_recovered: {e}")

    # =========================================================================
    # Monitoring Loop
    # =========================================================================

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        logger.info(
            f"[SystemHealthMonitor] Starting with check_interval="
            f"{self.config.check_interval_seconds}s"
        )

        while self._running:
            try:
                await self._check_health()
            except Exception as e:
                logger.error(f"[SystemHealthMonitor] Health check failed: {e}")

            await asyncio.sleep(self.config.check_interval_seconds)

    async def _check_health(self) -> None:
        """Perform a health check."""
        health = self.calculate_health_score()
        self._last_health_score = health
        self._last_check_time = time.time()

        # Update history
        self._health_history.append(health)
        if len(self._health_history) > self._max_history:
            self._health_history = self._health_history[-self._max_history :]

        # Log based on level
        if health.level == SystemHealthLevel.CRITICAL:
            logger.warning(
                f"[SystemHealthMonitor] CRITICAL - Score: {health.score}, "
                f"Components: {health.components}"
            )
        elif health.level == SystemHealthLevel.UNHEALTHY:
            logger.warning(
                f"[SystemHealthMonitor] Unhealthy - Score: {health.score}"
            )
        elif health.level == SystemHealthLevel.DEGRADED:
            logger.info(f"[SystemHealthMonitor] Degraded - Score: {health.score}")
        else:
            logger.debug(f"[SystemHealthMonitor] Healthy - Score: {health.score}")

        # Notify health callbacks
        for callback in self._health_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(health)
                else:
                    callback(health)
            except Exception as e:
                logger.debug(f"[SystemHealthMonitor] Health callback failed: {e}")

        # Check pause/resume
        if self._should_pause(health):
            await self._pause_pipeline(health)
        elif self._should_resume(health):
            await self._resume_pipeline(health)

    # =========================================================================
    # Public API
    # =========================================================================

    async def start(self) -> None:
        """Start the health monitor."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("[SystemHealthMonitor] Started")

    async def stop(self) -> None:
        """Stop the health monitor."""
        if not self._running:
            return

        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("[SystemHealthMonitor] Stopped")

    def get_health_score(self) -> int:
        """Get current health score (0-100)."""
        if self._last_health_score:
            return self._last_health_score.score
        return self.calculate_health_score().score

    def get_health_level(self) -> SystemHealthLevel:
        """Get current health level."""
        if self._last_health_score:
            return self._last_health_score.level
        return self.calculate_health_score().level

    def get_health_details(self) -> SystemHealthScore:
        """Get detailed health score."""
        if self._last_health_score:
            return self._last_health_score
        return self.calculate_health_score()

    def is_pipeline_paused(self) -> bool:
        """Check if pipeline is currently paused."""
        return self._pipeline_state == PipelineState.PAUSED

    def get_pipeline_state(self) -> PipelineState:
        """Get current pipeline state."""
        return self._pipeline_state

    def get_pause_reason(self) -> str:
        """Get reason for current pause."""
        return self._pause_reason

    def get_pause_duration(self) -> float:
        """Get duration of current pause in seconds."""
        if self._pipeline_state != PipelineState.PAUSED:
            return 0.0
        return time.time() - self._paused_at

    def get_health_history(self, limit: int = 60) -> list[SystemHealthScore]:
        """Get recent health history."""
        return self._health_history[-limit:]

    def get_status(self) -> dict[str, Any]:
        """Get monitor status for display."""
        health = self._last_health_score or self.calculate_health_score()

        return {
            "running": self._running,
            "health_score": health.score,
            "health_level": health.level.value,
            "components": health.components,
            "pipeline_state": self._pipeline_state.value,
            "pause_reason": self._pause_reason,
            "pause_duration_seconds": self.get_pause_duration(),
            "last_check": self._last_check_time,
            "history_size": len(self._health_history),
        }

    # =========================================================================
    # Callback Registration
    # =========================================================================

    def on_pause(self, callback: callable) -> None:
        """Register callback for pipeline pause."""
        self._pause_callbacks.append(callback)

    def on_resume(self, callback: callable) -> None:
        """Register callback for pipeline resume."""
        self._resume_callbacks.append(callback)

    def on_health_change(self, callback: callable) -> None:
        """Register callback for health score updates."""
        self._health_callbacks.append(callback)

    # =========================================================================
    # Manual Controls
    # =========================================================================

    async def force_pause(self, reason: str = "manual") -> None:
        """Manually pause the pipeline."""
        if self._pipeline_state == PipelineState.PAUSED:
            return

        self._pipeline_state = PipelineState.PAUSED
        self._paused_at = time.time()
        self._pause_reason = f"manual:{reason}"

        logger.warning(f"[SystemHealthMonitor] Pipeline manually paused: {reason}")

    async def force_resume(self) -> None:
        """Manually resume the pipeline."""
        if self._pipeline_state != PipelineState.PAUSED:
            return

        pause_duration = time.time() - self._paused_at

        logger.info(
            f"[SystemHealthMonitor] Pipeline manually resumed after {pause_duration:.0f}s"
        )

        self._pipeline_state = PipelineState.RUNNING
        self._paused_at = 0.0
        self._pause_reason = ""


# =============================================================================
# Singleton and Convenience Functions
# =============================================================================

_system_health_monitor: SystemHealthMonitorDaemon | None = None


def get_system_health() -> SystemHealthMonitorDaemon:
    """Get the global SystemHealthMonitorDaemon singleton."""
    global _system_health_monitor
    if _system_health_monitor is None:
        _system_health_monitor = SystemHealthMonitorDaemon()
    return _system_health_monitor


def is_pipeline_paused() -> bool:
    """Check if pipeline is paused due to system health."""
    return get_system_health().is_pipeline_paused()


def get_health_score() -> int:
    """Get current system health score (0-100)."""
    return get_system_health().get_health_score()


def reset_system_health_monitor() -> None:
    """Reset the singleton (for testing)."""
    global _system_health_monitor
    _system_health_monitor = None


__all__ = [
    # Enums
    "PipelineState",
    "SystemHealthLevel",
    # Data classes
    "SystemHealthConfig",
    "SystemHealthScore",
    # Main class
    "SystemHealthMonitorDaemon",
    # Functions
    "get_health_score",
    "get_system_health",
    "is_pipeline_paused",
    "reset_system_health_monitor",
]
