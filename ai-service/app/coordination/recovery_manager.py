"""
Centralized recovery coordination for self-healing cluster operations.

This module provides automatic recovery from stuck jobs, unhealthy nodes,
and other failure conditions without requiring manual intervention.

Extends CoordinatorBase for standardized lifecycle management and stats.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from app.coordination.coordinator_base import CoordinatorBase, CoordinatorStatus

if TYPE_CHECKING:
    from app.coordination.work_queue import WorkItem, WorkQueue

logger = logging.getLogger(__name__)


class RecoveryAction(str, Enum):
    """Types of recovery actions."""
    RESTART_JOB = "restart_job"
    KILL_JOB = "kill_job"
    RESTART_NODE_SERVICES = "restart_node_services"
    REBOOT_NODE = "reboot_node"
    REMOVE_NODE = "remove_node"
    ESCALATE_HUMAN = "escalate_human"
    NONE = "none"


class RecoveryResult(str, Enum):
    """Result of a recovery attempt."""
    SUCCESS = "success"
    FAILED = "failed"
    ESCALATED = "escalated"
    SKIPPED = "skipped"


@dataclass
class RecoveryConfig:
    """Configuration for recovery behavior."""

    # Stuck job detection
    stuck_job_timeout_multiplier: float = 1.5  # Kill jobs at N * expected timeout

    # Recovery attempt limits
    max_recovery_attempts_per_node: int = 3    # Max attempts before escalation
    max_recovery_attempts_per_job: int = 2     # Max attempts per job
    recovery_attempt_cooldown: int = 300       # 5 min between attempts per node

    # Escalation thresholds
    consecutive_failures_for_escalation: int = 3
    escalation_cooldown: int = 3600            # 1 hour between escalations

    # Node health thresholds
    node_unhealthy_after_failures: int = 3     # Mark unhealthy after N failures
    node_recovery_timeout: int = 120           # Timeout for node recovery ops

    # Enabled flag
    enabled: bool = True


@dataclass
class RecoveryEvent:
    """Record of a recovery attempt."""
    timestamp: float
    action: RecoveryAction
    target_type: str  # "job" or "node"
    target_id: str
    result: RecoveryResult
    reason: str
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class NodeRecoveryState:
    """Track recovery state for a node."""
    node_id: str
    recovery_attempts: int = 0
    last_attempt_time: float = 0.0
    consecutive_failures: int = 0
    is_escalated: bool = False
    last_escalation_time: float = 0.0


@dataclass
class JobRecoveryState:
    """Track recovery state for a job."""
    work_id: str
    recovery_attempts: int = 0
    last_attempt_time: float = 0.0


class RecoveryManager(CoordinatorBase):
    """
    Centralized recovery coordination for self-healing operations.

    Extends CoordinatorBase for standardized lifecycle management.

    Handles:
    - Stuck job detection and recovery
    - Unhealthy node recovery
    - Escalation to human operators when auto-recovery fails

    Dependencies (set via set_dependency or legacy setters):
    - work_queue: WorkQueue instance
    - notifier: Notification service
    - kill_job_callback: Callable for killing jobs
    - restart_services_callback: Callable for restarting node services
    - reboot_node_callback: Callable for rebooting nodes
    """

    def __init__(
        self,
        config: Optional[RecoveryConfig] = None,
        notifier: Optional[Any] = None,
    ):
        super().__init__(name="RecoveryManager")
        self.config = config or RecoveryConfig()

        # State tracking
        self._node_states: Dict[str, NodeRecoveryState] = {}
        self._job_states: Dict[str, JobRecoveryState] = {}
        self._recovery_history: List[RecoveryEvent] = []

        # Set initial dependencies if provided
        if notifier:
            self.set_dependency("notifier", notifier)

        # Mark as ready immediately (no async init needed)
        self._status = CoordinatorStatus.READY

    # Legacy setters - delegate to dependency injection
    def set_work_queue(self, work_queue: "WorkQueue") -> None:
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

    # Property accessors for dependencies
    @property
    def _work_queue(self) -> Optional["WorkQueue"]:
        return self.get_dependency("work_queue")

    @property
    def _notifier(self) -> Optional[Any]:
        return self.get_dependency("notifier")

    @property
    def _kill_job_callback(self) -> Optional[Callable]:
        return self.get_dependency("kill_job_callback")

    @property
    def _restart_services_callback(self) -> Optional[Callable]:
        return self.get_dependency("restart_services_callback")

    @property
    def _reboot_node_callback(self) -> Optional[Callable]:
        return self.get_dependency("reboot_node_callback")

    def _get_node_state(self, node_id: str) -> NodeRecoveryState:
        """Get or create node recovery state."""
        if node_id not in self._node_states:
            self._node_states[node_id] = NodeRecoveryState(node_id=node_id)
        return self._node_states[node_id]

    def _get_job_state(self, work_id: str) -> JobRecoveryState:
        """Get or create job recovery state."""
        if work_id not in self._job_states:
            self._job_states[work_id] = JobRecoveryState(work_id=work_id)
        return self._job_states[work_id]

    def _can_attempt_node_recovery(self, node_id: str) -> bool:
        """Check if we can attempt recovery on this node."""
        state = self._get_node_state(node_id)

        # Check if already escalated
        if state.is_escalated:
            # Check if escalation cooldown has passed
            if time.time() - state.last_escalation_time < self.config.escalation_cooldown:
                return False
            # Reset escalation state after cooldown
            state.is_escalated = False

        # Check attempt limit
        if state.recovery_attempts >= self.config.max_recovery_attempts_per_node:
            return False

        # Check cooldown
        if time.time() - state.last_attempt_time < self.config.recovery_attempt_cooldown:
            return False

        return True

    def _can_attempt_job_recovery(self, work_id: str) -> bool:
        """Check if we can attempt recovery on this job."""
        state = self._get_job_state(work_id)
        return state.recovery_attempts < self.config.max_recovery_attempts_per_job

    async def recover_stuck_job(
        self,
        work_item: "WorkItem",
        expected_timeout: float,
    ) -> RecoveryResult:
        """
        Attempt to recover a stuck job.

        Args:
            work_item: The stuck work item
            expected_timeout: Expected timeout in seconds

        Returns:
            RecoveryResult indicating success/failure/escalation
        """
        if not self.config.enabled:
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
            if self._kill_job_callback and node_id:
                await self._kill_job_callback(node_id, work_id)

            # Mark as failed in work queue
            if self._work_queue:
                self._work_queue.fail_work(work_id, "stuck_timeout_recovery")

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
        """
        Attempt to recover an unhealthy node.

        Args:
            node_id: The node to recover
            reason: Reason for recovery attempt

        Returns:
            RecoveryResult indicating success/failure/escalation
        """
        if not self.config.enabled:
            return RecoveryResult.SKIPPED

        logger.info(f"Attempting to recover unhealthy node {node_id}: {reason}")

        node_state = self._get_node_state(node_id)

        if not self._can_attempt_node_recovery(node_id):
            # Escalate to human
            await self._escalate_to_human(node_id, reason)
            return RecoveryResult.ESCALATED

        start_time = time.time()
        node_state.recovery_attempts += 1
        node_state.last_attempt_time = start_time

        try:
            # Try restarting services first
            if self._restart_services_callback:
                success = await asyncio.wait_for(
                    self._restart_services_callback(node_id),
                    timeout=self.config.node_recovery_timeout,
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

            # If service restart failed/unavailable, try more aggressive recovery
            node_state.consecutive_failures += 1

            # Check if we should escalate
            if node_state.consecutive_failures >= self.config.consecutive_failures_for_escalation:
                await self._escalate_to_human(node_id, f"{reason} - {node_state.consecutive_failures} consecutive failures")
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

        # Send notification
        if self._notifier:
            try:
                await self._notifier.send_escalation_alert(
                    target_id=target_id,
                    reason=reason,
                    recovery_attempts=self._node_states.get(target_id, NodeRecoveryState(target_id)).recovery_attempts,
                )
            except Exception as e:
                logger.error(f"Failed to send escalation notification: {e}")

    def _record_event(
        self,
        action: RecoveryAction,
        target_type: str,
        target_id: str,
        result: RecoveryResult,
        reason: str,
        error: Optional[str] = None,
        duration: float = 0.0,
    ) -> None:
        """Record a recovery event."""
        event = RecoveryEvent(
            timestamp=time.time(),
            action=action,
            target_type=target_type,
            target_id=target_id,
            result=result,
            reason=reason,
            error=error,
            duration_seconds=duration,
        )
        self._recovery_history.append(event)

        # Keep last 500 events
        if len(self._recovery_history) > 500:
            self._recovery_history = self._recovery_history[-500:]

    def reset_node_state(self, node_id: str) -> None:
        """Reset recovery state for a node (e.g., after successful operation)."""
        if node_id in self._node_states:
            self._node_states[node_id] = NodeRecoveryState(node_id=node_id)

    def reset_job_state(self, work_id: str) -> None:
        """Reset recovery state for a job."""
        if work_id in self._job_states:
            del self._job_states[work_id]

    async def get_stats(self) -> Dict[str, Any]:
        """Get recovery statistics for monitoring.

        Implements CoordinatorBase.get_stats() interface.
        """
        # Get base stats from CoordinatorBase
        base_stats = await super().get_stats()

        recent_events = [e for e in self._recovery_history if time.time() - e.timestamp < 3600]

        success_count = sum(1 for e in recent_events if e.result == RecoveryResult.SUCCESS)
        failed_count = sum(1 for e in recent_events if e.result == RecoveryResult.FAILED)
        escalated_count = sum(1 for e in recent_events if e.result == RecoveryResult.ESCALATED)

        escalated_nodes = [
            node_id for node_id, state in self._node_states.items()
            if state.is_escalated
        ]

        # Merge with recovery-specific stats
        base_stats.update({
            "enabled": self.config.enabled,
            "recoveries_last_hour": {
                "success": success_count,
                "failed": failed_count,
                "escalated": escalated_count,
            },
            "nodes_tracked": len(self._node_states),
            "jobs_tracked": len(self._job_states),
            "escalated_nodes": escalated_nodes,
            "total_events": len(self._recovery_history),
        })
        return base_stats

    def get_recovery_stats(self) -> Dict[str, Any]:
        """Legacy sync wrapper for get_stats().

        Deprecated: Use await get_stats() instead.
        """
        import asyncio
        try:
            asyncio.get_running_loop()
            # If there's a running loop, we can't use run_until_complete
            # Return a simplified version
            return self._get_recovery_stats_sync()
        except RuntimeError:
            # No running loop, safe to create one
            return asyncio.run(self.get_stats())

    def _get_recovery_stats_sync(self) -> Dict[str, Any]:
        """Synchronous version of recovery stats (no base stats)."""
        recent_events = [e for e in self._recovery_history if time.time() - e.timestamp < 3600]

        success_count = sum(1 for e in recent_events if e.result == RecoveryResult.SUCCESS)
        failed_count = sum(1 for e in recent_events if e.result == RecoveryResult.FAILED)
        escalated_count = sum(1 for e in recent_events if e.result == RecoveryResult.ESCALATED)

        escalated_nodes = [
            node_id for node_id, state in self._node_states.items()
            if state.is_escalated
        ]

        return {
            "name": self.name,
            "status": self.status.value,
            "enabled": self.config.enabled,
            "recoveries_last_hour": {
                "success": success_count,
                "failed": failed_count,
                "escalated": escalated_count,
            },
            "nodes_tracked": len(self._node_states),
            "jobs_tracked": len(self._job_states),
            "escalated_nodes": escalated_nodes,
            "total_events": len(self._recovery_history),
        }

    def find_stuck_jobs(
        self,
        running_items: List["WorkItem"],
        timeout_multiplier: Optional[float] = None,
    ) -> List[tuple["WorkItem", float]]:
        """
        Find jobs that appear to be stuck.

        Args:
            running_items: List of currently running work items
            timeout_multiplier: Override for stuck detection (default from config)

        Returns:
            List of (work_item, expected_timeout) tuples for stuck jobs
        """
        multiplier = timeout_multiplier or self.config.stuck_job_timeout_multiplier
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


def load_recovery_config_from_yaml(yaml_config: Dict[str, Any]) -> RecoveryConfig:
    """Load RecoveryConfig from YAML configuration dict."""
    self_healing = yaml_config.get("self_healing", {})

    return RecoveryConfig(
        enabled=self_healing.get("enabled", True),
        stuck_job_timeout_multiplier=self_healing.get("stuck_job_timeout_multiplier", 1.5),
        max_recovery_attempts_per_node=self_healing.get("max_recovery_attempts_per_node", 3),
        max_recovery_attempts_per_job=self_healing.get("max_recovery_attempts_per_job", 2),
        recovery_attempt_cooldown=self_healing.get("recovery_attempt_cooldown", 300),
        consecutive_failures_for_escalation=self_healing.get("consecutive_failures_for_escalation", 3),
        escalation_cooldown=self_healing.get("escalation_cooldown", 3600),
        node_unhealthy_after_failures=self_healing.get("node_unhealthy_after_failures", 3),
        node_recovery_timeout=self_healing.get("node_recovery_timeout", 120),
    )


# =============================================================================
# Singleton and Event Wiring (December 2025)
# =============================================================================

_recovery_manager: Optional[RecoveryManager] = None


def get_recovery_manager() -> RecoveryManager:
    """Get the global RecoveryManager singleton."""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = RecoveryManager()
    return _recovery_manager


def wire_recovery_events() -> RecoveryManager:
    """Wire recovery events to the manager.

    Subscribes to:
    - TASK_FAILED: Track failed tasks for recovery
    - HOST_OFFLINE: Track offline hosts for recovery
    - ERROR: Track errors for recovery decisions
    - NODE_RECOVERED: Update recovery state

    Returns:
        The configured RecoveryManager
    """
    manager = get_recovery_manager()

    try:
        from app.distributed.data_events import DataEventType, get_event_bus

        bus = get_event_bus()

        async def _on_task_failed(event):
            """Handle TASK_FAILED event."""
            payload = event.payload
            work_id = payload.get("work_id") or payload.get("task_id", "")
            if work_id:
                state = manager._get_job_state(work_id)
                state.recovery_attempts += 1
                state.last_attempt_time = time.time()
                logger.debug(f"[RecoveryManager] Tracked failed task: {work_id}")

        async def _on_host_offline(event):
            """Handle HOST_OFFLINE event."""
            payload = event.payload
            node_id = payload.get("node_id") or payload.get("host_id", "")
            if node_id:
                state = manager._get_node_state(node_id)
                state.consecutive_failures += 1
                state.last_failure_time = time.time()
                logger.debug(f"[RecoveryManager] Tracked offline host: {node_id}")

        async def _on_node_recovered(event):
            """Handle NODE_RECOVERED event."""
            payload = event.payload
            node_id = payload.get("node_id") or payload.get("host_id", "")
            if node_id and node_id in manager._node_states:
                state = manager._node_states[node_id]
                state.consecutive_failures = 0
                state.is_unhealthy = False
                logger.debug(f"[RecoveryManager] Node recovered: {node_id}")

        bus.subscribe(DataEventType.TASK_FAILED, _on_task_failed)
        bus.subscribe(DataEventType.HOST_OFFLINE, _on_host_offline)
        bus.subscribe(DataEventType.NODE_RECOVERED, _on_node_recovered)

        logger.info("[RecoveryManager] Subscribed to recovery events")

    except ImportError:
        logger.warning("[RecoveryManager] data_events not available")
    except Exception as e:
        logger.error(f"[RecoveryManager] Failed to subscribe to events: {e}")

    return manager


__all__ = [
    # Enums
    "RecoveryAction",
    "RecoveryResult",
    # Data classes
    "RecoveryConfig",
    "RecoveryEvent",
    "NodeRecoveryState",
    "JobRecoveryState",
    # Main class
    "RecoveryManager",
    # Functions
    "load_recovery_config_from_yaml",
    "get_recovery_manager",
    "wire_recovery_events",
]
