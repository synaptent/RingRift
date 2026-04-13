"""Recovery, query, and scoring helpers for UnifiedHealthManager."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from app.coordination.unified_health_shared import (
    CircuitState,
    ErrorRecord,
    ErrorSeverity,
    HAS_NODE_EVENTS,
    HealthStats,
    NodeHealthState,
    RecoveryAction,
    RecoveryAttempt,
    RecoveryResult,
    RecoveryStatus,
    SystemHealthConfig,
    SystemHealthLevel,
    SystemHealthScore,
    emit_node_overloaded,
)

if TYPE_CHECKING:
    from app.coordination.work_queue import WorkItem

logger = logging.getLogger(__name__)


class UnifiedHealthRecoveryMixin:
    """Extracted helpers for UnifiedHealthManager."""

    async def recover_stuck_job(
        self,
        work_item: "WorkItem",
        expected_timeout: float,
    ) -> RecoveryResult:
        """Attempt to recover a stuck job.

        Args:
            work_item: The stuck work item
            expected_timeout: Expected timeout in seconds

        Returns:
            RecoveryResult indicating success/failure/escalation
        """
        if not self._health_config.enabled:
            return RecoveryResult.SKIPPED

        work_id = work_item.work_id
        node_id = work_item.claimed_by

        logger.info(f"Attempting to recover stuck job {work_id} on node {node_id}")

        job_state = self._get_job_state(work_id)

        if not self._can_attempt_job_recovery(work_id):
            logger.warning(f"Max recovery attempts reached for job {work_id}")
            return RecoveryResult.ESCALATED

        start_time = time.time()
        job_state.recovery_attempts += 1
        job_state.last_attempt_time = start_time

        try:
            # Kill the job on the node
            kill_callback = self.get_dependency("kill_job_callback")
            if kill_callback and node_id:
                await kill_callback(node_id, work_id)

            # Mark as failed in work queue
            work_queue = self.get_dependency("work_queue")
            if work_queue:
                work_queue.fail_work(work_id, "stuck_timeout_recovery")

            duration = time.time() - start_time
            self._record_event(
                action=RecoveryAction.KILL_JOB,
                target_type="job",
                target_id=work_id,
                result=RecoveryResult.SUCCESS,
                reason=f"job_stuck_exceeded_{expected_timeout}s",
                duration=duration,
            )

            logger.info(f"Successfully recovered stuck job {work_id}")
            return RecoveryResult.SUCCESS

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed to recover stuck job {work_id}: {e}")

            self._record_event(
                action=RecoveryAction.KILL_JOB,
                target_type="job",
                target_id=work_id,
                result=RecoveryResult.FAILED,
                reason=f"job_stuck_exceeded_{expected_timeout}s",
                error=str(e),
                duration=duration,
            )

            return RecoveryResult.FAILED

    async def recover_unhealthy_node(self, node_id: str, reason: str) -> RecoveryResult:
        """Attempt to recover an unhealthy node.

        Args:
            node_id: The node to recover
            reason: Reason for recovery attempt

        Returns:
            RecoveryResult indicating success/failure/escalation
        """
        if not self._health_config.enabled:
            return RecoveryResult.SKIPPED

        logger.info(f"Attempting to recover unhealthy node {node_id}: {reason}")

        node_state = self._get_node_state(node_id)

        if not self._can_attempt_node_recovery(node_id):
            await self._escalate_to_human(node_id, reason)
            return RecoveryResult.ESCALATED

        start_time = time.time()
        node_state.recovery_attempts += 1
        node_state.last_attempt_time = start_time

        try:
            # Try restarting services first
            restart_callback = self.get_dependency("restart_services_callback")
            if restart_callback:
                success = await asyncio.wait_for(
                    restart_callback(node_id),
                    timeout=self._health_config.node_recovery_timeout,
                )

                if success:
                    duration = time.time() - start_time
                    node_state.consecutive_failures = 0

                    self._record_event(
                        action=RecoveryAction.RESTART_NODE_SERVICES,
                        target_type="node",
                        target_id=node_id,
                        result=RecoveryResult.SUCCESS,
                        reason=reason,
                        duration=duration,
                    )

                    logger.info(f"Successfully recovered node {node_id} via service restart")
                    return RecoveryResult.SUCCESS

            # If service restart failed, increment failure count
            node_state.consecutive_failures += 1

            # Check if we should escalate
            if node_state.consecutive_failures >= self._health_config.consecutive_failures_for_escalation:
                await self._escalate_to_human(
                    node_id, f"{reason} - {node_state.consecutive_failures} consecutive failures"
                )
                return RecoveryResult.ESCALATED

            duration = time.time() - start_time
            self._record_event(
                action=RecoveryAction.RESTART_NODE_SERVICES,
                target_type="node",
                target_id=node_id,
                result=RecoveryResult.FAILED,
                reason=reason,
                duration=duration,
            )

            return RecoveryResult.FAILED

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            node_state.consecutive_failures += 1

            logger.error(f"Recovery timeout for node {node_id}")

            self._record_event(
                action=RecoveryAction.RESTART_NODE_SERVICES,
                target_type="node",
                target_id=node_id,
                result=RecoveryResult.FAILED,
                reason=reason,
                error="timeout",
                duration=duration,
            )

            return RecoveryResult.FAILED

        except Exception as e:
            duration = time.time() - start_time
            node_state.consecutive_failures += 1

            logger.error(f"Failed to recover node {node_id}: {e}")

            self._record_event(
                action=RecoveryAction.RESTART_NODE_SERVICES,
                target_type="node",
                target_id=node_id,
                result=RecoveryResult.FAILED,
                reason=reason,
                error=str(e),
                duration=duration,
            )

            return RecoveryResult.FAILED

    async def _escalate_to_human(self, target_id: str, reason: str) -> None:
        """Escalate an issue to human operators."""
        logger.warning(f"Escalating to human: {target_id} - {reason}")

        if target_id in self._node_states:
            state = self._node_states[target_id]
            state.is_escalated = True
            state.last_escalation_time = time.time()

        self._record_event(
            action=RecoveryAction.ESCALATE_HUMAN,
            target_type="node" if target_id in self._node_states else "job",
            target_id=target_id,
            result=RecoveryResult.ESCALATED,
            reason=reason,
        )

        # Emit NODE_OVERLOADED event for job redistribution (Phase 21.2 - Dec 2025)
        if HAS_NODE_EVENTS and emit_node_overloaded and target_id in self._node_states:
            try:
                state = self._node_states[target_id]
                await emit_node_overloaded(
                    host=target_id,
                    cpu_percent=100.0,  # Assume overloaded when escalated
                    gpu_percent=0.0,  # No GPU info available from health manager
                    memory_percent=0.0,
                    resource_type="consecutive_failures",
                    source="unified_health_manager.py",
                )
                logger.info(f"[UnifiedHealthManager] Emitted NODE_OVERLOADED for {target_id}")
            except Exception as e:
                logger.debug(f"[UnifiedHealthManager] Failed to emit NODE_OVERLOADED: {e}")

        # Notify escalation callbacks
        for callback in self._escalation_callbacks:
            try:
                callback(target_id, reason)
            except Exception as e:
                logger.error(f"[UnifiedHealthManager] Escalation callback failed: {e}")

        # Send notification
        notifier = self.get_dependency("notifier")
        if notifier:
            try:
                await notifier.send_escalation_alert(
                    target_id=target_id,
                    reason=reason,
                    recovery_attempts=self._node_states.get(
                        target_id, NodeHealthState(target_id)
                    ).recovery_attempts,
                )
            except Exception as e:
                logger.error(f"Failed to send escalation notification: {e}")

    def find_stuck_jobs(
        self,
        running_items: list["WorkItem"],
        timeout_multiplier: float | None = None,
    ) -> list[tuple["WorkItem", float]]:
        """Find jobs that appear to be stuck.

        Args:
            running_items: List of currently running work items
            timeout_multiplier: Override for stuck detection

        Returns:
            List of (work_item, expected_timeout) tuples for stuck jobs
        """
        multiplier = timeout_multiplier or self._health_config.stuck_job_timeout_multiplier
        stuck_jobs = []
        current_time = time.time()

        for item in running_items:
            expected_timeout = item.timeout_seconds
            actual_runtime = current_time - item.started_at if item.started_at else 0

            if actual_runtime > expected_timeout * multiplier:
                stuck_jobs.append((item, expected_timeout))
                logger.debug(
                    f"Detected stuck job {item.work_id}: "
                    f"runtime={actual_runtime:.0f}s > expected={expected_timeout * multiplier:.0f}s"
                )

        return stuck_jobs

    def record_error(
        self,
        component: str,
        error_type: str,
        message: str,
        node_id: str = "",
        severity: str = "error",
        context: dict | None = None,
    ) -> ErrorRecord:
        """Manually record an error.

        Returns:
            The created ErrorRecord
        """
        error = ErrorRecord(
            error_id=self._generate_error_id(),
            component=component,
            error_type=error_type,
            message=message,
            node_id=node_id,
            severity=ErrorSeverity(severity),
            context=context or {},
        )

        self._record_error(error)
        return error

    def start_recovery(
        self,
        error_id: str,
        component: str,
        node_id: str,
        strategy: str = "default",
    ) -> RecoveryAttempt:
        """Start a recovery attempt.

        Returns:
            The created RecoveryAttempt
        """
        recovery = RecoveryAttempt(
            recovery_id=self._generate_recovery_id(),
            error_id=error_id,
            component=component,
            node_id=node_id,
            strategy=strategy,
            status=RecoveryStatus.IN_PROGRESS,
        )

        self._active_recoveries[recovery.recovery_id] = recovery
        self._total_recoveries += 1

        return recovery

    def complete_recovery(self, recovery_id: str, success: bool, message: str = "") -> None:
        """Complete a recovery attempt."""
        if recovery_id not in self._active_recoveries:
            return

        recovery = self._active_recoveries.pop(recovery_id)
        recovery.status = RecoveryStatus.COMPLETED if success else RecoveryStatus.FAILED
        recovery.completed_at = time.time()
        recovery.success = success
        recovery.message = message

        self._record_recovery(recovery)

        if success:
            self._successful_recoveries += 1
            self._total_recovery_time += recovery.duration
            self._on_component_success(recovery.component)
        else:
            self._failed_recoveries += 1
            self._on_component_failure(recovery.component)

    def on_error(self, callback: Callable[[ErrorRecord], None]) -> None:
        """Register callback for errors."""
        self._error_callbacks.append(callback)

    def on_recovery(self, callback: Callable[[RecoveryAttempt], None]) -> None:
        """Register callback for recovery completions."""
        self._recovery_callbacks.append(callback)

    def on_circuit_breaker_change(self, callback: Callable[[str, bool], None]) -> None:
        """Register callback for circuit breaker state changes."""
        self._circuit_breaker_callbacks.append(callback)

    def on_escalation(self, callback: Callable[[str, str], None]) -> None:
        """Register callback for escalations."""
        self._escalation_callbacks.append(callback)

    def set_work_queue(self, work_queue: "WorkItem") -> None:
        """Set the work queue reference."""
        self.set_dependency("work_queue", work_queue)

    def set_notifier(self, notifier: Any) -> None:
        """Set the notification service."""
        self.set_dependency("notifier", notifier)

    def set_kill_job_callback(self, callback: Callable) -> None:
        """Set callback for killing jobs."""
        self.set_dependency("kill_job_callback", callback)

    def set_restart_services_callback(self, callback: Callable) -> None:
        """Set callback for restarting node services."""
        self.set_dependency("restart_services_callback", callback)

    def set_reboot_node_callback(self, callback: Callable) -> None:
        """Set callback for rebooting nodes."""
        self.set_dependency("reboot_node_callback", callback)

    def reset_node_state(self, node_id: str) -> None:
        """Reset health state for a node."""
        if node_id in self._node_states:
            self._node_states[node_id] = NodeHealthState(node_id=node_id)

    def reset_job_state(self, work_id: str) -> None:
        """Reset health state for a job."""
        if work_id in self._job_states:
            del self._job_states[work_id]

    def get_recent_errors(self, limit: int = 50) -> list[ErrorRecord]:
        """Get recent errors."""
        return self._errors[-limit:]

    def get_errors_by_component(self, component: str) -> list[ErrorRecord]:
        """Get errors for a specific component."""
        return list(self._errors_by_component.get(component, []))

    def get_recovery_history(self, limit: int = 50) -> list[RecoveryAttempt]:
        """Get recent recovery attempts."""
        return self._recovery_history[-limit:]

    def get_active_recoveries(self) -> list[RecoveryAttempt]:
        """Get active recovery attempts."""
        return list(self._active_recoveries.values())

    def get_circuit_breaker_states(self) -> dict[str, CircuitState]:
        """Get all circuit breaker states."""
        return {
            component: cb.get_state(component) for component, cb in self._circuit_breakers.items()
        }

    def get_online_nodes(self) -> set[str]:
        """Get set of online nodes."""
        return {
            node_id
            for node_id, state in self._node_states.items()
            if state.is_online
        }

    def get_offline_nodes(self) -> dict[str, float]:
        """Get offline nodes with offline timestamp."""
        return {
            node_id: state.offline_since
            for node_id, state in self._node_states.items()
            if not state.is_online and state.offline_since > 0
        }

    def get_escalated_nodes(self) -> list[str]:
        """Get list of escalated nodes."""
        return [
            node_id
            for node_id, state in self._node_states.items()
            if state.is_escalated
        ]

    def get_health_stats(self) -> HealthStats:
        """Get aggregate health statistics."""
        # Count by severity
        by_severity: dict[str, int] = defaultdict(int)
        for error in self._errors:
            by_severity[error.severity.value] += 1

        # Count by component
        by_component: dict[str, int] = {
            comp: len(errors) for comp, errors in self._errors_by_component.items()
        }

        # Count by node
        by_node: dict[str, int] = defaultdict(int)
        for error in self._errors:
            if error.node_id:
                by_node[error.node_id] += 1

        # Recovery rate
        recovery_rate = (
            self._successful_recoveries / self._total_recoveries
            if self._total_recoveries > 0
            else 0.0
        )

        # Average recovery time
        avg_recovery_time = (
            self._total_recovery_time / self._successful_recoveries
            if self._successful_recoveries > 0
            else 0.0
        )

        # Circuit breaker stats
        open_circuits = [
            comp for comp, cb in self._circuit_breakers.items() if cb.get_state(comp) == CircuitState.OPEN
        ]

        # Node stats
        online_nodes = sum(1 for s in self._node_states.values() if s.is_online)
        offline_nodes = sum(1 for s in self._node_states.values() if not s.is_online)

        return HealthStats(
            total_errors=self._total_errors,
            errors_by_severity=dict(by_severity),
            errors_by_component=by_component,
            errors_by_node=dict(by_node),
            recovery_attempts=self._total_recoveries,
            successful_recoveries=self._successful_recoveries,
            failed_recoveries=self._failed_recoveries,
            recovery_rate=recovery_rate,
            avg_recovery_time=avg_recovery_time,
            circuit_breakers_open=len(open_circuits),
            open_circuits=open_circuits,
            nodes_tracked=len(self._node_states),
            nodes_online=online_nodes,
            nodes_offline=offline_nodes,
            escalated_nodes=self.get_escalated_nodes(),
            jobs_tracked=len(self._job_states),
        )

    async def get_stats(self) -> dict[str, Any]:
        """Get recovery statistics for monitoring.

        Implements CoordinatorBase.get_stats() interface.
        """
        base_stats = super().get_stats()
        health_stats = self.get_health_stats()

        recent_events = [
            e for e in self._recovery_events if time.time() - e.timestamp < 3600
        ]
        success_count = sum(1 for e in recent_events if e.result == RecoveryResult.SUCCESS)
        failed_count = sum(1 for e in recent_events if e.result == RecoveryResult.FAILED)
        escalated_count = sum(1 for e in recent_events if e.result == RecoveryResult.ESCALATED)

        base_stats.update({
            "enabled": self._health_config.enabled,
            "total_errors": health_stats.total_errors,
            "errors_by_severity": health_stats.errors_by_severity,
            "errors_by_component": health_stats.errors_by_component,
            "recovery_attempts": health_stats.recovery_attempts,
            "successful_recoveries": health_stats.successful_recoveries,
            "failed_recoveries": health_stats.failed_recoveries,
            "recovery_rate": round(health_stats.recovery_rate * 100, 1),
            "avg_recovery_time": round(health_stats.avg_recovery_time, 1),
            "active_recoveries": len(self._active_recoveries),
            "circuit_breakers_open": health_stats.circuit_breakers_open,
            "open_circuits": health_stats.open_circuits,
            "recoveries_last_hour": {
                "success": success_count,
                "failed": failed_count,
                "escalated": escalated_count,
            },
            "nodes_tracked": health_stats.nodes_tracked,
            "nodes_online": health_stats.nodes_online,
            "nodes_offline": health_stats.nodes_offline,
            "escalated_nodes": health_stats.escalated_nodes,
            "jobs_tracked": health_stats.jobs_tracked,
            "subscribed": self._stats.subscribed,
        })
        return base_stats

    def calculate_system_health_score(
        self, sys_config: SystemHealthConfig | None = None
    ) -> SystemHealthScore:
        """Calculate aggregate system health score.

        Consolidated from system_health_monitor.py - now directly available
        on the health manager.

        Args:
            sys_config: Optional system health config, uses defaults if not provided

        Returns:
            SystemHealthScore with aggregate health data
        """
        cfg = sys_config or SystemHealthConfig()

        # Calculate component scores
        node_availability = self._calculate_node_availability(cfg)
        circuit_health = self._calculate_circuit_health(cfg)
        error_rate = self._calculate_error_rate_score(cfg)
        recovery_success = self._calculate_recovery_success_score()

        # Weighted aggregate
        score = (
            node_availability * cfg.node_weight
            + circuit_health * cfg.circuit_weight
            + error_rate * cfg.error_weight
            + recovery_success * cfg.recovery_weight
        )

        score = int(max(0, min(100, score)))

        # Determine level
        if score >= cfg.healthy_threshold:
            level = SystemHealthLevel.HEALTHY
        elif score >= cfg.degraded_threshold:
            level = SystemHealthLevel.DEGRADED
        elif score >= cfg.unhealthy_threshold:
            level = SystemHealthLevel.UNHEALTHY
        else:
            level = SystemHealthLevel.CRITICAL

        # Check pause triggers
        pause_triggers = self._check_pause_triggers(
            cfg, score, node_availability, circuit_health, error_rate
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

    def _calculate_node_availability(self, cfg: SystemHealthConfig) -> float:
        """Calculate node availability score (0-100).

        Feb 26, 2026: When no nodes are tracked (e.g., master_loop process
        which doesn't receive P2P heartbeat events), return 100% instead of 0%.
        The absence of node data should not be treated as "all nodes offline"
        since that triggers false TRAINING_BLOCKED events.
        """
        nodes_tracked = len(self._node_states)
        if nodes_tracked == 0:
            # No data about nodes — assume healthy rather than pausing pipeline
            return 100.0

        nodes_online = sum(1 for s in self._node_states.values() if s.is_online)

        # Determine expected nodes
        expected = cfg.expected_nodes
        if expected == 0:
            expected = max(nodes_tracked, 1)

        # Calculate availability
        availability = (nodes_online / expected) * 100 if expected > 0 else 100.0
        return min(100.0, availability)

    def _calculate_circuit_health(self, cfg: SystemHealthConfig) -> float:
        """Calculate circuit breaker health score (0-100)."""
        total_circuits = len(self._circuit_breakers)
        if total_circuits == 0:
            return 100.0

        open_circuits = sum(
            1
            for comp, cb in self._circuit_breakers.items()
            if cb.get_state(comp) == CircuitState.OPEN
        )

        # Circuits closed percentage
        closed_percent = ((total_circuits - open_circuits) / total_circuits) * 100

        # Extra penalty for critical circuits
        critical_open = [
            c
            for c in cfg.critical_circuits
            if c in self._circuit_breakers
            and self._circuit_breakers[c].get_state(c) == CircuitState.OPEN
        ]

        if critical_open:
            # Heavy penalty for critical circuits
            penalty = len(critical_open) * 20
            closed_percent = max(0, closed_percent - penalty)

        return closed_percent

    def _calculate_error_rate_score(self, cfg: SystemHealthConfig) -> float:
        """Calculate error rate score (0-100, inverted: higher = fewer errors)."""
        # Check recent errors
        now = time.time()
        window = cfg.pause_error_burst_window
        recent_errors = [e for e in self._errors if now - e.timestamp < window]

        error_count = len(recent_errors)

        # Score based on error count
        threshold = cfg.pause_error_burst_count
        if error_count >= threshold:
            return 0.0

        score = ((threshold - error_count) / threshold) * 100
        return max(0.0, min(100.0, score))

    def _calculate_recovery_success_score(self) -> float:
        """Calculate recovery success rate (0-100)."""
        total = self._total_recoveries
        successful = self._successful_recoveries

        if total == 0:
            return 100.0  # No recoveries needed = healthy

        return (successful / total) * 100

    def _check_pause_triggers(
        self,
        cfg: SystemHealthConfig,
        score: int,
        node_availability: float,
        circuit_health: float,
        error_rate: float,
    ) -> list[str]:
        """Check for conditions that should trigger pipeline pause."""
        triggers = []

        # Health score threshold
        if score < cfg.pause_health_threshold:
            triggers.append(f"health_score_critical:{score}")

        # Node offline threshold
        offline_percent = (100 - node_availability) / 100
        if offline_percent >= cfg.pause_node_offline_percent:
            triggers.append(f"nodes_offline:{offline_percent:.0%}")

        # Critical circuit broken
        for circuit_name in cfg.critical_circuits:
            if circuit_name in self._circuit_breakers:
                cb = self._circuit_breakers[circuit_name]
                if cb.get_state(circuit_name) == CircuitState.OPEN:
                    triggers.append(f"critical_circuit_open:{circuit_name}")

        # Error burst
        if error_rate == 0:
            triggers.append("error_burst_detected")

        return triggers
