"""Budget-aware capacity planning for cluster management.

This module tracks spending across providers and provides scaling
recommendations based on utilization and budget constraints.

Created: Dec 28, 2025
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from app.coordination.base_daemon import BaseDaemon, DaemonConfig

if TYPE_CHECKING:
    from app.coordination.providers.base import Instance

logger = logging.getLogger(__name__)


# December 2025: Import from canonical source to avoid collision with resource_optimizer.py
from app.coordination.enums import ScaleAction


@dataclass
class CapacityBudget:
    """Budget constraints for cluster capacity.

    Tracks hourly and daily spending limits along with
    current accumulated costs.
    """
    hourly_limit_usd: float = 50.0  # $50/hour max
    daily_limit_usd: float = 500.0  # $500/day max
    alert_threshold_percent: float = 80.0  # Alert at 80% budget

    # Tracked spending (reset hourly/daily)
    current_hourly_usd: float = 0.0
    current_daily_usd: float = 0.0

    # Timestamps for reset
    hourly_reset_time: datetime = field(default_factory=datetime.now)
    daily_reset_time: datetime = field(default_factory=datetime.now)

    def remaining_hourly_budget(self) -> float:
        """Get remaining hourly budget."""
        return max(0.0, self.hourly_limit_usd - self.current_hourly_usd)

    def remaining_daily_budget(self) -> float:
        """Get remaining daily budget."""
        return max(0.0, self.daily_limit_usd - self.current_daily_usd)

    def hourly_budget_percent_used(self) -> float:
        """Get hourly budget usage as percentage."""
        if self.hourly_limit_usd <= 0:
            return 100.0
        return (self.current_hourly_usd / self.hourly_limit_usd) * 100

    def daily_budget_percent_used(self) -> float:
        """Get daily budget usage as percentage."""
        if self.daily_limit_usd <= 0:
            return 100.0
        return (self.current_daily_usd / self.daily_limit_usd) * 100

    def is_over_alert_threshold(self) -> bool:
        """Check if spending is above alert threshold."""
        return (
            self.hourly_budget_percent_used() >= self.alert_threshold_percent
            or self.daily_budget_percent_used() >= self.alert_threshold_percent
        )

    def can_afford(self, additional_hourly_cost: float) -> bool:
        """Check if we can afford additional hourly spend."""
        # Check both hourly and daily limits
        if self.current_hourly_usd + additional_hourly_cost > self.hourly_limit_usd:
            return False

        # Estimate daily impact (assume running for remaining hours)
        hours_in_day = 24
        daily_impact = additional_hourly_cost * hours_in_day
        if self.current_daily_usd + daily_impact > self.daily_limit_usd:
            return False

        return True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "hourly_limit_usd": self.hourly_limit_usd,
            "daily_limit_usd": self.daily_limit_usd,
            "current_hourly_usd": self.current_hourly_usd,
            "current_daily_usd": self.current_daily_usd,
            "hourly_percent_used": self.hourly_budget_percent_used(),
            "daily_percent_used": self.daily_budget_percent_used(),
            "alert_threshold_percent": self.alert_threshold_percent,
        }


@dataclass
class ScaleRecommendation:
    """Scaling recommendation from CapacityPlanner."""
    action: ScaleAction
    count: int = 0
    reason: str = ""
    provider: str = ""
    gpu_type: str = ""
    estimated_hourly_cost: float = 0.0
    utilization: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for event emission."""
        return {
            "action": self.action.value,
            "count": self.count,
            "reason": self.reason,
            "provider": self.provider,
            "gpu_type": self.gpu_type,
            "estimated_hourly_cost": self.estimated_hourly_cost,
            "utilization": self.utilization,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(kw_only=True)
class CapacityPlannerConfig(DaemonConfig):
    """Configuration for CapacityPlanner."""
    check_interval_seconds: int = 60  # 1 minute (overrides DaemonConfig default)

    # Utilization thresholds
    scale_up_utilization_threshold: float = 0.85  # Scale up at 85% utilization
    scale_down_utilization_threshold: float = 0.30  # Scale down below 30%

    # Minimum stability period before scaling
    min_scale_up_interval_seconds: float = 300.0  # 5 minutes between scale-ups
    min_scale_down_interval_seconds: float = 600.0  # 10 minutes between scale-downs

    # Budget limits
    hourly_budget_usd: float = 50.0
    daily_budget_usd: float = 500.0

    # Minimum capacity
    min_gpu_nodes: int = 4

    # GPU cost estimates ($/hour)
    gpu_costs: dict = field(default_factory=lambda: {
        "GH200_96GB": 2.49,
        "H100_80GB": 2.49,
        "A100_80GB": 1.29,
        "A100_40GB": 1.10,
        "A10": 0.60,
        "RTX_4090": 0.80,
        "RTX_3090": 0.50,
    })

    @classmethod
    def from_env(cls) -> "CapacityPlannerConfig":
        """Load configuration from environment variables."""
        return cls(
            check_interval_seconds=int(
                os.environ.get("RINGRIFT_CAPACITY_CYCLE_INTERVAL", "60")
            ),
            scale_up_utilization_threshold=float(
                os.environ.get("RINGRIFT_SCALE_UP_THRESHOLD", "0.85")
            ),
            scale_down_utilization_threshold=float(
                os.environ.get("RINGRIFT_SCALE_DOWN_THRESHOLD", "0.30")
            ),
            hourly_budget_usd=float(
                os.environ.get("RINGRIFT_HOURLY_BUDGET_USD", "50")
            ),
            daily_budget_usd=float(
                os.environ.get("RINGRIFT_DAILY_BUDGET_USD", "500")
            ),
            min_gpu_nodes=int(
                os.environ.get("RINGRIFT_MIN_GPU_NODES", "4")
            ),
        )


@dataclass
class UtilizationMetrics:
    """Cluster utilization metrics."""
    total_gpu_nodes: int = 0
    active_gpu_nodes: int = 0
    gpu_utilization_avg: float = 0.0
    memory_utilization_avg: float = 0.0
    selfplay_jobs_running: int = 0
    training_jobs_running: int = 0
    pending_work_items: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def overall_utilization(self) -> float:
        """Calculate overall cluster utilization."""
        if self.total_gpu_nodes == 0:
            return 0.0

        # Weight GPU utilization highest, then jobs, then memory
        job_util = min(1.0, (self.selfplay_jobs_running + self.training_jobs_running) / max(1, self.active_gpu_nodes))

        return (
            0.5 * self.gpu_utilization_avg
            + 0.3 * job_util
            + 0.2 * self.memory_utilization_avg
        )


class CapacityPlanner(BaseDaemon):
    """Budget-aware capacity planner for cluster management.

    Monitors cluster utilization and spending, providing scaling
    recommendations that respect budget constraints.

    Example:
        planner = CapacityPlanner()
        await planner.start()

        # Check if we can scale up
        if await planner.should_scale_up(2):
            # Provision 2 nodes
            pass

        # Get recommendation
        rec = await planner.get_scale_recommendation()
        if rec.action == ScaleAction.SCALE_UP:
            # Handle scale up
            pass
    """

    def __init__(self, config: CapacityPlannerConfig | None = None):
        super().__init__(config)

        self.budget = CapacityBudget(
            hourly_limit_usd=self.config.hourly_budget_usd,
            daily_limit_usd=self.config.daily_budget_usd,
        )

        self._last_scale_up_time: float = 0.0
        self._last_scale_down_time: float = 0.0
        self._utilization_history: list[UtilizationMetrics] = []
        self._cost_history: list[tuple[datetime, float]] = []

    def _get_default_config(self) -> CapacityPlannerConfig:
        """Return default configuration."""
        return CapacityPlannerConfig()

    def _get_daemon_name(self) -> str:
        """Return daemon name for logging."""
        return "CapacityPlanner"

    def _get_event_subscriptions(self) -> dict:
        """Subscribe to cost-related events."""
        return {
            "NODE_PROVISIONED": self._on_node_provisioned,
            "NODE_TERMINATED": self._on_node_terminated,
            "HOURLY_TICK": self._on_hourly_tick,
        }

    async def _on_node_provisioned(self, event: dict) -> None:
        """Track costs when nodes are provisioned."""
        payload = event.get("payload", event)
        cost_per_hour = payload.get("cost_per_hour", 0.0)

        self.budget.current_hourly_usd += cost_per_hour
        self.budget.current_daily_usd += cost_per_hour

        self._cost_history.append((datetime.now(), cost_per_hour))

        logger.info(
            f"CapacityPlanner: Added ${cost_per_hour:.2f}/hr to budget. "
            f"Now: ${self.budget.current_hourly_usd:.2f}/hr hourly, "
            f"${self.budget.current_daily_usd:.2f} daily"
        )

    async def _on_node_terminated(self, event: dict) -> None:
        """Reduce tracked costs when nodes are terminated."""
        payload = event.get("payload", event)
        cost_per_hour = payload.get("cost_per_hour", 0.0)

        self.budget.current_hourly_usd = max(
            0.0, self.budget.current_hourly_usd - cost_per_hour
        )

        logger.info(
            f"CapacityPlanner: Removed ${cost_per_hour:.2f}/hr from budget. "
            f"Now: ${self.budget.current_hourly_usd:.2f}/hr"
        )

    async def _on_hourly_tick(self, event: dict) -> None:
        """Reset hourly budget counter."""
        self.budget.hourly_reset_time = datetime.now()
        self.budget.current_hourly_usd = await self._calculate_current_hourly_cost()

        logger.debug(
            f"CapacityPlanner: Hourly reset. Current cost: "
            f"${self.budget.current_hourly_usd:.2f}/hr"
        )

    async def _run_cycle(self) -> None:
        """Run one capacity planning cycle."""
        # Reset daily budget if needed
        now = datetime.now()
        if now.date() > self.budget.daily_reset_time.date():
            self.budget.daily_reset_time = now
            self.budget.current_daily_usd = 0.0
            logger.info("CapacityPlanner: Daily budget reset")

        # Collect utilization metrics
        metrics = await self._collect_utilization_metrics()
        self._utilization_history.append(metrics)

        # Keep only recent history (1 hour)
        cutoff = now - timedelta(hours=1)
        self._utilization_history = [
            m for m in self._utilization_history
            if m.timestamp > cutoff
        ]

        # Check for budget alerts
        if self.budget.is_over_alert_threshold():
            await self._emit_budget_alert()

        # Log current status
        logger.debug(
            f"CapacityPlanner: util={metrics.overall_utilization:.1%}, "
            f"nodes={metrics.active_gpu_nodes}/{metrics.total_gpu_nodes}, "
            f"budget=${self.budget.current_hourly_usd:.2f}/hr"
        )

    async def _collect_utilization_metrics(self) -> UtilizationMetrics:
        """Collect current cluster utilization metrics."""
        metrics = UtilizationMetrics()

        try:
            # Get node counts from cluster config
            from app.config.cluster_config import get_cluster_nodes

            nodes = get_cluster_nodes()

            for node_id, node in nodes.items():
                if getattr(node, "is_gpu_node", False):
                    metrics.total_gpu_nodes += 1
                    if getattr(node, "is_active", True):
                        metrics.active_gpu_nodes += 1

            # Get GPU utilization from node monitor
            try:
                from .node_monitor import get_node_monitor

                monitor = get_node_monitor()
                statuses = monitor.get_all_node_statuses()

                gpu_utils = []
                mem_utils = []

                for node_id, status in statuses.items():
                    if status.get("healthy", False):
                        gpu_util = status.get("gpu_utilization", 0.0)
                        mem_util = status.get("memory_utilization", 0.0)
                        if gpu_util > 0:
                            gpu_utils.append(gpu_util)
                        if mem_util > 0:
                            mem_utils.append(mem_util)

                if gpu_utils:
                    metrics.gpu_utilization_avg = sum(gpu_utils) / len(gpu_utils)
                if mem_utils:
                    metrics.memory_utilization_avg = sum(mem_utils) / len(mem_utils)

            except Exception as e:
                logger.warning(f"Failed to get GPU utilization: {e}")

            # Get job counts from P2P status
            try:
                import aiohttp
                from app.config.ports import get_p2p_status_url

                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        get_p2p_status_url(),
                        timeout=aiohttp.ClientTimeout(total=5.0),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            metrics.selfplay_jobs_running = data.get("selfplay_jobs", 0)
                            metrics.training_jobs_running = data.get("training_jobs", 0)
                            metrics.pending_work_items = data.get("pending_work", 0)

            except Exception as e:
                logger.debug(f"Failed to get P2P status: {e}")

        except Exception as e:
            logger.error(f"Failed to collect utilization metrics: {e}")

        return metrics

    async def _calculate_current_hourly_cost(self) -> float:
        """Calculate current hourly cost from running instances."""
        total_cost = 0.0

        try:
            from app.coordination.providers.registry import get_all_providers

            for provider in get_all_providers():
                if not provider.is_configured():
                    continue

                try:
                    instances = await provider.list_instances()
                    for inst in instances:
                        if inst.status.value == "running":
                            total_cost += inst.cost_per_hour
                except Exception as e:
                    logger.warning(f"Failed to get instances from {provider.name}: {e}")

        except ImportError:
            # Provider registry not available
            pass

        return total_cost

    async def should_scale_up(self, count: int = 1) -> bool:
        """Check if we can afford to scale up.

        Args:
            count: Number of nodes to add

        Returns:
            True if budget allows scaling up
        """
        # Estimate cost for new nodes (use average GPU cost)
        avg_cost = sum(self.config.gpu_costs.values()) / len(self.config.gpu_costs)
        additional_cost = avg_cost * count

        # Check budget
        if not self.budget.can_afford(additional_cost):
            logger.info(
                f"CapacityPlanner: Cannot afford {count} node(s) at "
                f"${additional_cost:.2f}/hr (budget: "
                f"${self.budget.remaining_hourly_budget():.2f}/hr remaining)"
            )
            return False

        # Check scale-up cooldown
        if time.time() - self._last_scale_up_time < self.config.min_scale_up_interval_seconds:
            logger.debug("CapacityPlanner: Scale-up cooldown active")
            return False

        return True

    async def get_scale_recommendation(self) -> ScaleRecommendation:
        """Get scaling recommendation based on utilization and budget.

        Returns:
            ScaleRecommendation with action, count, and details
        """
        metrics = await self._collect_utilization_metrics()
        utilization = metrics.overall_utilization

        # Check if we need to scale up
        if utilization >= self.config.scale_up_utilization_threshold:
            # High utilization - recommend scale up
            if await self.should_scale_up(2):
                return ScaleRecommendation(
                    action=ScaleAction.SCALE_UP,
                    count=2,
                    reason=f"High utilization ({utilization:.1%} >= {self.config.scale_up_utilization_threshold:.1%})",
                    provider="lambda",  # Prefer Lambda for GH200
                    gpu_type="GH200_96GB",
                    estimated_hourly_cost=self.config.gpu_costs.get("GH200_96GB", 2.49) * 2,
                    utilization=utilization,
                )
            else:
                return ScaleRecommendation(
                    action=ScaleAction.NONE,
                    count=0,
                    reason=f"High utilization but budget exceeded",
                    utilization=utilization,
                )

        # Check if we can scale down
        if (
            utilization <= self.config.scale_down_utilization_threshold
            and metrics.active_gpu_nodes > self.config.min_gpu_nodes
        ):
            # Check scale-down cooldown
            if time.time() - self._last_scale_down_time >= self.config.min_scale_down_interval_seconds:
                return ScaleRecommendation(
                    action=ScaleAction.SCALE_DOWN,
                    count=1,
                    reason=f"Low utilization ({utilization:.1%} <= {self.config.scale_down_utilization_threshold:.1%})",
                    utilization=utilization,
                )

        # No scaling needed
        return ScaleRecommendation(
            action=ScaleAction.NONE,
            count=0,
            reason="Utilization within acceptable range",
            utilization=utilization,
        )

    def record_scale_up(self) -> None:
        """Record that a scale-up occurred (for cooldown tracking)."""
        self._last_scale_up_time = time.time()

    def record_scale_down(self) -> None:
        """Record that a scale-down occurred (for cooldown tracking)."""
        self._last_scale_down_time = time.time()

    async def _emit_budget_alert(self) -> None:
        """Emit budget alert event."""
        await self._safe_emit_event(
            "BUDGET_ALERT",
            {
                "hourly_percent_used": self.budget.hourly_budget_percent_used(),
                "daily_percent_used": self.budget.daily_budget_percent_used(),
                "hourly_limit": self.budget.hourly_limit_usd,
                "daily_limit": self.budget.daily_limit_usd,
                "current_hourly": self.budget.current_hourly_usd,
                "current_daily": self.budget.current_daily_usd,
                "threshold": self.budget.alert_threshold_percent,
            },
        )

    async def _safe_emit_event(self, event_type: str, payload: dict) -> None:
        """Safely emit an event via the event router."""
        try:
            from app.coordination.event_router import get_event_bus

            bus = get_event_bus()
            if bus:
                from app.distributed.data_events import DataEvent

                event = DataEvent(
                    event_type=event_type,
                    payload=payload,
                    source="CapacityPlanner",
                )
                bus.publish(event)
        except Exception as e:
            logger.debug(f"Event emission failed: {e}")

    def get_budget_status(self) -> dict:
        """Get current budget status."""
        return self.budget.to_dict()

    def get_utilization_history(self, limit: int = 60) -> list[dict]:
        """Get recent utilization history."""
        return [
            {
                "timestamp": m.timestamp.isoformat(),
                "total_gpu_nodes": m.total_gpu_nodes,
                "active_gpu_nodes": m.active_gpu_nodes,
                "gpu_utilization": m.gpu_utilization_avg,
                "memory_utilization": m.memory_utilization_avg,
                "overall_utilization": m.overall_utilization,
                "selfplay_jobs": m.selfplay_jobs_running,
                "training_jobs": m.training_jobs_running,
            }
            for m in self._utilization_history[-limit:]
        ]

    def health_check(self) -> dict:
        """Return health status for DaemonManager integration."""
        budget_ok = not self.budget.is_over_alert_threshold()

        recent_metrics = (
            self._utilization_history[-1] if self._utilization_history else None
        )

        return {
            "healthy": budget_ok,
            "message": (
                "CapacityPlanner: Budget within limits"
                if budget_ok
                else f"CapacityPlanner: Budget at {self.budget.hourly_budget_percent_used():.0f}% of limit"
            ),
            "details": {
                "hourly_budget_percent": self.budget.hourly_budget_percent_used(),
                "daily_budget_percent": self.budget.daily_budget_percent_used(),
                "current_hourly_usd": self.budget.current_hourly_usd,
                "active_gpu_nodes": (
                    recent_metrics.active_gpu_nodes if recent_metrics else 0
                ),
                "overall_utilization": (
                    recent_metrics.overall_utilization if recent_metrics else 0.0
                ),
            },
        }


# Singleton instance
_capacity_planner: CapacityPlanner | None = None


def get_capacity_planner() -> CapacityPlanner:
    """Get the singleton CapacityPlanner instance."""
    global _capacity_planner
    if _capacity_planner is None:
        _capacity_planner = CapacityPlanner()
    return _capacity_planner


def reset_capacity_planner() -> None:
    """Reset the singleton (for testing)."""
    global _capacity_planner
    _capacity_planner = None
