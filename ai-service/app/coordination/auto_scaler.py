"""
Auto-scaling integration for Vast.ai instances based on work queue depth.

This module provides queue-depth-based scaling that automatically provisions
or deprovisions Vast.ai GPU instances based on pending work and idle nodes.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.coordination.work_queue import WorkQueue

logger = logging.getLogger(__name__)


class ScalingAction(str, Enum):
    """Scaling action types."""
    NONE = "none"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling behavior."""

    # Queue depth thresholds
    queue_depth_scale_up: int = 10      # Scale up if > N pending items
    queue_depth_scale_down: int = 2     # Scale down if < N pending items

    # GPU idle detection
    gpu_idle_minutes: int = 15          # Consider node idle after N minutes < 10% util
    gpu_idle_threshold_percent: float = 10.0  # GPU utilization threshold for idle

    # Instance limits
    min_instances: int = 2              # Always keep at least N instances
    max_instances: int = 20             # Cap for cost control

    # Cost control
    max_hourly_cost: float = 2.00       # Budget cap in $/hour

    # Cooldown to prevent thrashing
    scale_cooldown_seconds: int = 600   # 10 minutes between scale operations

    # Scaling increments
    max_scale_up_per_cycle: int = 3     # Max instances to add per cycle
    max_scale_down_per_cycle: int = 2   # Max instances to remove per cycle

    # Predictive scaling
    predictive_scaling: bool = True     # Enable demand prediction
    prediction_hours_ahead: int = 2     # Hours ahead to predict

    # Enabled flag
    enabled: bool = True


@dataclass
class ScalingDecision:
    """Result of scaling evaluation."""
    action: ScalingAction
    count: int = 0
    node_ids: List[str] = field(default_factory=list)
    reason: str = ""
    estimated_cost_change: float = 0.0


@dataclass
class ScaleEvent:
    """Record of a scaling event for history tracking."""
    timestamp: float
    action: ScalingAction
    count: int
    node_ids: List[str]
    reason: str
    success: bool
    error: Optional[str] = None


@dataclass
class NodeMetrics:
    """Metrics for a single node."""
    node_id: str
    gpu_utilization: float
    last_job_time: float
    hourly_cost: float
    instance_type: str
    is_idle: bool = False


class AutoScaler:
    """
    Queue-based auto-scaler for Vast.ai GPU instances.

    Monitors work queue depth and GPU utilization to make scaling decisions.
    Integrates with the P2P orchestrator's loop system.
    """

    def __init__(
        self,
        config: Optional[ScalingConfig] = None,
        vast_client: Optional[Any] = None,  # VastClient when available
    ):
        self.config = config or ScalingConfig()
        self.vast_client = vast_client

        # State tracking
        self._last_scale_time: float = 0
        self._scale_history: List[ScaleEvent] = []
        self._active_instances: Dict[str, NodeMetrics] = {}
        self._node_idle_since: Dict[str, float] = {}  # node_id -> timestamp

        # Cost tracking
        self._cost_start_time: float = time.time()
        self._cost_samples: List[tuple] = []  # (timestamp, hourly_cost)
        self._total_cost_accumulated: float = 0.0  # Cumulative cost in $
        self._last_cost_update: float = time.time()

        # Reference to work queue (set by orchestrator)
        self._work_queue: Optional["WorkQueue"] = None

    def set_work_queue(self, work_queue: "WorkQueue") -> None:
        """Set the work queue reference."""
        self._work_queue = work_queue

    def set_vast_client(self, vast_client: Any) -> None:
        """Set the Vast.ai client reference."""
        self.vast_client = vast_client

    def update_node_metrics(self, node_id: str, metrics: NodeMetrics) -> None:
        """Update metrics for a node."""
        self._active_instances[node_id] = metrics

        # Update cumulative cost
        self._update_cumulative_cost()

        # Track idle state transitions
        if metrics.is_idle:
            if node_id not in self._node_idle_since:
                self._node_idle_since[node_id] = time.time()
        else:
            self._node_idle_since.pop(node_id, None)

    def remove_node(self, node_id: str) -> None:
        """Remove a node from tracking."""
        self._active_instances.pop(node_id, None)
        self._node_idle_since.pop(node_id, None)

    def get_pending_count(self) -> int:
        """Get count of pending work items."""
        if self._work_queue is None:
            return 0
        status = self._work_queue.get_queue_status()
        return status.get("by_status", {}).get("pending", 0)

    def get_running_count(self) -> int:
        """Get count of running work items."""
        if self._work_queue is None:
            return 0
        status = self._work_queue.get_queue_status()
        by_status = status.get("by_status", {})
        return by_status.get("running", 0) + by_status.get("claimed", 0)

    def _get_idle_nodes(self) -> List[str]:
        """Get list of nodes that have been idle beyond the threshold."""
        idle_threshold = time.time() - (self.config.gpu_idle_minutes * 60)
        idle_nodes = []

        for node_id, idle_since in self._node_idle_since.items():
            if idle_since < idle_threshold:
                idle_nodes.append(node_id)

        # Sort by idle duration (longest idle first)
        idle_nodes.sort(key=lambda n: self._node_idle_since.get(n, time.time()))
        return idle_nodes

    def _get_current_hourly_cost(self) -> float:
        """Get current total hourly cost of all instances."""
        return sum(m.hourly_cost for m in self._active_instances.values())

    def _update_cumulative_cost(self) -> None:
        """Update cumulative cost based on time elapsed and current hourly rate."""
        now = time.time()
        elapsed_hours = (now - self._last_cost_update) / 3600.0
        current_hourly = self._get_current_hourly_cost()

        # Accumulate cost
        if elapsed_hours > 0:
            self._total_cost_accumulated += current_hourly * elapsed_hours
            self._cost_samples.append((now, current_hourly))

            # Keep only last 24 hours of samples
            cutoff = now - 86400
            self._cost_samples = [(t, c) for t, c in self._cost_samples if t > cutoff]

        self._last_cost_update = now

    def get_cost_metrics(self) -> Dict[str, Any]:
        """Get cost tracking metrics.

        Returns:
            Dictionary with cost metrics including:
            - current_hourly_cost: Current $/hour burn rate
            - total_cost_accumulated: Total $ spent since tracking started
            - tracking_hours: Hours since tracking started
            - average_hourly_cost: Average $/hour over tracking period
            - cost_last_24h: Estimated cost in last 24 hours
            - projected_monthly: Projected monthly cost at current rate
        """
        self._update_cumulative_cost()
        now = time.time()

        tracking_hours = (now - self._cost_start_time) / 3600.0
        current_hourly = self._get_current_hourly_cost()

        # Calculate 24h cost from samples
        cutoff_24h = now - 86400
        samples_24h = [(t, c) for t, c in self._cost_samples if t > cutoff_24h]
        cost_last_24h = 0.0
        if len(samples_24h) >= 2:
            for i in range(1, len(samples_24h)):
                t1, c1 = samples_24h[i - 1]
                t2, _ = samples_24h[i]
                hours = (t2 - t1) / 3600.0
                cost_last_24h += c1 * hours

        # Average hourly cost
        avg_hourly = 0.0
        if tracking_hours > 0:
            avg_hourly = self._total_cost_accumulated / tracking_hours

        return {
            "current_hourly_cost": round(current_hourly, 4),
            "total_cost_accumulated": round(self._total_cost_accumulated, 2),
            "tracking_hours": round(tracking_hours, 2),
            "average_hourly_cost": round(avg_hourly, 4),
            "cost_last_24h": round(cost_last_24h, 2),
            "projected_daily": round(current_hourly * 24, 2),
            "projected_monthly": round(current_hourly * 24 * 30, 2),
            "budget_hourly": self.config.max_hourly_cost,
            "budget_remaining_hourly": round(self.config.max_hourly_cost - current_hourly, 4),
            "budget_utilization_pct": round(current_hourly / max(self.config.max_hourly_cost, 0.01) * 100, 1),
        }

    def _get_instance_count(self) -> int:
        """Get current number of active instances."""
        return len(self._active_instances)

    def _is_in_cooldown(self) -> bool:
        """Check if we're in cooldown period."""
        return time.time() - self._last_scale_time < self.config.scale_cooldown_seconds

    def _predict_demand(self, hours_ahead: int = 2) -> int:
        """
        Predict work demand based on historical patterns.

        Returns estimated number of pending items in the future.
        """
        if not self.config.predictive_scaling:
            return self.get_pending_count()

        # Simple prediction based on current queue growth rate
        # In production, this would analyze historical patterns
        current_pending = self.get_pending_count()
        current_running = self.get_running_count()

        # Estimate based on running jobs completing and new jobs arriving
        # This is a placeholder - real implementation would use ML or time series
        estimated_completion_rate = current_running / max(1, hours_ahead)
        estimated_arrival_rate = current_pending / max(1, hours_ahead)

        predicted = current_pending + (estimated_arrival_rate - estimated_completion_rate) * hours_ahead
        return max(0, int(predicted))

    async def evaluate(self) -> ScalingDecision:
        """
        Evaluate whether to scale up or down.

        Returns a ScalingDecision with the recommended action.
        """
        if not self.config.enabled:
            return ScalingDecision(
                action=ScalingAction.NONE,
                reason="auto_scaling_disabled",
            )

        # Check cooldown
        if self._is_in_cooldown():
            cooldown_remaining = self.config.scale_cooldown_seconds - (
                time.time() - self._last_scale_time
            )
            return ScalingDecision(
                action=ScalingAction.NONE,
                reason=f"cooldown_active_{int(cooldown_remaining)}s_remaining",
            )

        pending = self.get_pending_count()
        running = self.get_running_count()
        idle_nodes = self._get_idle_nodes()
        current_cost = self._get_current_hourly_cost()
        instance_count = self._get_instance_count()

        logger.debug(
            f"AutoScaler evaluate: pending={pending}, running={running}, "
            f"idle_nodes={len(idle_nodes)}, instances={instance_count}, "
            f"cost=${current_cost:.2f}/hr"
        )

        # Check for scale up
        if pending > self.config.queue_depth_scale_up:
            # Calculate how many instances to add
            work_per_instance = max(1, (pending + running) / max(1, instance_count))
            instances_needed = int((pending - self.config.queue_depth_scale_up) / work_per_instance)
            instances_to_add = min(
                instances_needed,
                self.config.max_scale_up_per_cycle,
                self.config.max_instances - instance_count,
            )

            # Check cost budget
            avg_cost_per_instance = current_cost / max(1, instance_count) if instance_count > 0 else 0.5
            projected_cost = current_cost + (instances_to_add * avg_cost_per_instance)

            if projected_cost > self.config.max_hourly_cost:
                affordable = int((self.config.max_hourly_cost - current_cost) / max(0.1, avg_cost_per_instance))
                instances_to_add = max(0, min(instances_to_add, affordable))

            if instances_to_add > 0:
                return ScalingDecision(
                    action=ScalingAction.SCALE_UP,
                    count=instances_to_add,
                    reason=f"queue_depth_{pending}_exceeds_threshold_{self.config.queue_depth_scale_up}",
                    estimated_cost_change=instances_to_add * avg_cost_per_instance,
                )

        # Check for scale down
        if pending < self.config.queue_depth_scale_down and idle_nodes:
            # Don't scale below minimum
            max_removable = instance_count - self.config.min_instances
            if max_removable <= 0:
                return ScalingDecision(
                    action=ScalingAction.NONE,
                    reason=f"at_min_instances_{self.config.min_instances}",
                )

            nodes_to_remove = min(
                len(idle_nodes),
                self.config.max_scale_down_per_cycle,
                max_removable,
            )

            if nodes_to_remove > 0:
                removal_candidates = idle_nodes[:nodes_to_remove]
                estimated_savings = sum(
                    self._active_instances.get(n, NodeMetrics(n, 0, 0, 0.5, "")).hourly_cost
                    for n in removal_candidates
                )

                return ScalingDecision(
                    action=ScalingAction.SCALE_DOWN,
                    count=nodes_to_remove,
                    node_ids=removal_candidates,
                    reason=f"queue_depth_{pending}_below_threshold_{self.config.queue_depth_scale_down}_with_{len(idle_nodes)}_idle",
                    estimated_cost_change=-estimated_savings,
                )

        return ScalingDecision(
            action=ScalingAction.NONE,
            reason=f"within_bounds_pending_{pending}_idle_{len(idle_nodes)}",
        )

    def record_scale_event(
        self,
        decision: ScalingDecision,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Record a scaling event for history tracking."""
        event = ScaleEvent(
            timestamp=time.time(),
            action=decision.action,
            count=decision.count,
            node_ids=decision.node_ids,
            reason=decision.reason,
            success=success,
            error=error,
        )
        self._scale_history.append(event)

        # Keep last 100 events
        if len(self._scale_history) > 100:
            self._scale_history = self._scale_history[-100:]

        if success:
            self._last_scale_time = time.time()

        logger.info(
            f"Scale event: action={decision.action.value}, count={decision.count}, "
            f"success={success}, reason={decision.reason}"
        )

    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics for monitoring."""
        recent_events = [e for e in self._scale_history if time.time() - e.timestamp < 3600]

        scale_up_count = sum(1 for e in recent_events if e.action == ScalingAction.SCALE_UP and e.success)
        scale_down_count = sum(1 for e in recent_events if e.action == ScalingAction.SCALE_DOWN and e.success)
        failed_count = sum(1 for e in recent_events if not e.success)

        # Get cost metrics
        cost_metrics = self.get_cost_metrics()

        return {
            "enabled": self.config.enabled,
            "current_instances": self._get_instance_count(),
            "min_instances": self.config.min_instances,
            "max_instances": self.config.max_instances,
            "pending_work": self.get_pending_count(),
            "running_work": self.get_running_count(),
            "idle_nodes": len(self._get_idle_nodes()),
            "in_cooldown": self._is_in_cooldown(),
            "last_scale_time": self._last_scale_time,
            "scale_up_last_hour": scale_up_count,
            "scale_down_last_hour": scale_down_count,
            "failed_scales_last_hour": failed_count,
            # Cost tracking
            **cost_metrics,
        }


# =============================================================================
# Monitoring→AutoScaler Event Integration (December 2025)
# Connects UnifiedClusterMonitor alerts to AutoScaler for reactive scaling
# =============================================================================


class MonitoringAwareAutoScaler(AutoScaler):
    """AutoScaler with event-driven monitoring integration.

    Extends AutoScaler to react to monitoring alerts:
    - RESOURCE_CONSTRAINT: Pause scale-up if resources are constrained
    - NODE_UNHEALTHY: Track unhealthy nodes for scale-down decisions
    - P2P_CLUSTER_UNHEALTHY: Conservative scaling during cluster issues
    - HEALTH_ALERT: Log and track for scaling decisions

    Usage:
        from app.coordination.auto_scaler import MonitoringAwareAutoScaler

        scaler = MonitoringAwareAutoScaler()
        scaler.subscribe_to_monitoring_events()

        # Now scaling decisions will incorporate cluster health
        decision = await scaler.evaluate()
    """

    def __init__(
        self,
        config: Optional[ScalingConfig] = None,
        vast_client: Optional[Any] = None,
    ):
        super().__init__(config, vast_client)

        # Monitoring state (December 2025)
        self._monitoring_alerts: Dict[str, float] = {}  # alert_key -> timestamp
        self._unhealthy_nodes: Dict[str, float] = {}    # node_id -> timestamp
        self._cluster_healthy: bool = True
        self._resource_constrained: bool = False
        self._event_subscribed: bool = False

    def subscribe_to_monitoring_events(self) -> bool:
        """Subscribe to monitoring events from UnifiedClusterMonitor.

        Returns:
            True if successfully subscribed
        """
        try:
            from app.distributed.data_events import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()

            # Subscribe to relevant monitoring events
            bus.subscribe(DataEventType.RESOURCE_CONSTRAINT, self._on_resource_constraint)
            bus.subscribe(DataEventType.NODE_UNHEALTHY, self._on_node_unhealthy)
            bus.subscribe(DataEventType.P2P_CLUSTER_UNHEALTHY, self._on_cluster_unhealthy)
            bus.subscribe(DataEventType.P2P_CLUSTER_HEALTHY, self._on_cluster_healthy)
            bus.subscribe(DataEventType.HEALTH_ALERT, self._on_health_alert)

            self._event_subscribed = True
            logger.info("[MonitoringAwareAutoScaler] Subscribed to monitoring events")
            return True

        except ImportError:
            logger.warning("[MonitoringAwareAutoScaler] Event bus not available")
            return False
        except Exception as e:
            logger.warning(f"[MonitoringAwareAutoScaler] Failed to subscribe: {e}")
            return False

    def _on_resource_constraint(self, event) -> None:
        """Handle RESOURCE_CONSTRAINT event - pause scale-up."""
        payload = event.payload if hasattr(event, 'payload') else {}
        alert = payload.get("alert", "resource constraint")

        self._resource_constrained = True
        self._monitoring_alerts[f"resource:{alert}"] = time.time()

        logger.warning(
            f"[MonitoringAwareAutoScaler] Resource constraint detected: {alert}. "
            f"Scale-up paused."
        )

    def _on_node_unhealthy(self, event) -> None:
        """Handle NODE_UNHEALTHY event - track for scale-down."""
        payload = event.payload if hasattr(event, 'payload') else {}
        node_name = payload.get("node_name", "")

        if node_name:
            self._unhealthy_nodes[node_name] = time.time()
            logger.warning(
                f"[MonitoringAwareAutoScaler] Node {node_name} marked unhealthy. "
                f"Total unhealthy: {len(self._unhealthy_nodes)}"
            )

    def _on_cluster_unhealthy(self, event) -> None:
        """Handle P2P_CLUSTER_UNHEALTHY event - conservative scaling."""
        self._cluster_healthy = False
        logger.warning(
            "[MonitoringAwareAutoScaler] Cluster unhealthy. "
            "Entering conservative scaling mode."
        )

    def _on_cluster_healthy(self, event) -> None:
        """Handle P2P_CLUSTER_HEALTHY event - resume normal scaling."""
        self._cluster_healthy = True
        self._resource_constrained = False

        # Clear stale alerts (older than 5 minutes)
        cutoff = time.time() - 300
        self._monitoring_alerts = {
            k: v for k, v in self._monitoring_alerts.items() if v > cutoff
        }
        self._unhealthy_nodes = {
            k: v for k, v in self._unhealthy_nodes.items() if v > cutoff
        }

        logger.info(
            "[MonitoringAwareAutoScaler] Cluster healthy. "
            "Resuming normal scaling mode."
        )

    def _on_health_alert(self, event) -> None:
        """Handle HEALTH_ALERT event - log and track."""
        payload = event.payload if hasattr(event, 'payload') else {}
        alert = payload.get("alert", "unknown")

        self._monitoring_alerts[f"health:{alert}"] = time.time()
        logger.debug(f"[MonitoringAwareAutoScaler] Health alert: {alert}")

    async def evaluate(self) -> ScalingDecision:
        """Evaluate scaling decision with monitoring awareness.

        Incorporates cluster health and resource constraints into decisions.
        """
        # Check if resource constrained - block scale-up
        if self._resource_constrained:
            return ScalingDecision(
                action=ScalingAction.NONE,
                reason="resource_constrained_by_monitoring",
            )

        # Get base scaling decision
        decision = await super().evaluate()

        # If cluster is unhealthy, be conservative
        if not self._cluster_healthy:
            if decision.action == ScalingAction.SCALE_UP:
                # Reduce scale-up during cluster issues
                reduced_count = max(1, decision.count // 2)
                return ScalingDecision(
                    action=ScalingAction.SCALE_UP,
                    count=reduced_count,
                    reason=f"{decision.reason}_reduced_cluster_unhealthy",
                    estimated_cost_change=decision.estimated_cost_change * (reduced_count / max(1, decision.count)),
                )

        # If we have unhealthy nodes, prioritize them for scale-down
        if decision.action == ScalingAction.SCALE_DOWN and self._unhealthy_nodes:
            unhealthy_list = list(self._unhealthy_nodes.keys())
            overlap = [n for n in decision.node_ids if n in unhealthy_list]

            if overlap:
                # Prioritize unhealthy nodes
                return ScalingDecision(
                    action=ScalingAction.SCALE_DOWN,
                    count=len(overlap),
                    node_ids=overlap,
                    reason=f"{decision.reason}_prioritizing_unhealthy",
                    estimated_cost_change=decision.estimated_cost_change,
                )

        return decision

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring integration status."""
        return {
            "event_subscribed": self._event_subscribed,
            "cluster_healthy": self._cluster_healthy,
            "resource_constrained": self._resource_constrained,
            "unhealthy_nodes": list(self._unhealthy_nodes.keys()),
            "active_alerts": list(self._monitoring_alerts.keys()),
        }


def load_scaling_config_from_yaml(yaml_config: Dict[str, Any]) -> ScalingConfig:
    """Load ScalingConfig from YAML configuration dict."""
    auto_scaling = yaml_config.get("auto_scaling", {})

    return ScalingConfig(
        enabled=auto_scaling.get("enabled", True),
        queue_depth_scale_up=auto_scaling.get("queue_depth_scale_up", 10),
        queue_depth_scale_down=auto_scaling.get("queue_depth_scale_down", 2),
        gpu_idle_minutes=auto_scaling.get("gpu_idle_minutes", 15),
        gpu_idle_threshold_percent=auto_scaling.get("gpu_idle_threshold_percent", 10.0),
        min_instances=auto_scaling.get("min_instances", 2),
        max_instances=auto_scaling.get("max_instances", 20),
        max_hourly_cost=auto_scaling.get("max_hourly_cost", 2.00),
        scale_cooldown_seconds=auto_scaling.get("scale_cooldown_seconds", 600),
        max_scale_up_per_cycle=auto_scaling.get("max_scale_up_per_cycle", 3),
        max_scale_down_per_cycle=auto_scaling.get("max_scale_down_per_cycle", 2),
        predictive_scaling=auto_scaling.get("predictive_scaling", True),
        prediction_hours_ahead=auto_scaling.get("prediction_hours_ahead", 2),
    )


def create_monitoring_aware_scaler(
    config: Optional[ScalingConfig] = None,
    vast_client: Optional[Any] = None,
    subscribe_events: bool = True,
) -> MonitoringAwareAutoScaler:
    """Create an AutoScaler with monitoring event integration.

    Args:
        config: Scaling configuration
        vast_client: Vast.ai client for instance management
        subscribe_events: Whether to subscribe to monitoring events

    Returns:
        MonitoringAwareAutoScaler instance
    """
    scaler = MonitoringAwareAutoScaler(config, vast_client)

    if subscribe_events:
        scaler.subscribe_to_monitoring_events()

    return scaler


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Enums
    "ScalingAction",
    # Data classes
    "ScalingConfig",
    "ScalingDecision",
    "ScaleEvent",
    "NodeMetrics",
    # Classes
    "AutoScaler",
    "MonitoringAwareAutoScaler",
    # Functions
    "load_scaling_config_from_yaml",
    "create_monitoring_aware_scaler",
]
