"""Status-report helpers for coordination package consumers."""

from __future__ import annotations

from app.coordination.cache_coordination_orchestrator import get_cache_orchestrator
from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator
from app.coordination.event_router import get_coordinator_stats as get_event_coordinator_stats
from app.coordination.metrics_analysis_orchestrator import get_metrics_orchestrator
from app.coordination.optimization_coordinator import get_optimization_coordinator
from app.coordination.resource_monitoring_coordinator import get_resource_coordinator
from app.coordination.selfplay_orchestrator import get_selfplay_orchestrator
from app.coordination.task_lifecycle_coordinator import get_task_lifecycle_coordinator

__all__ = ["get_all_coordinator_status"]


def get_all_coordinator_status() -> dict:
    """Get unified status from all orchestrators and coordinators."""

    return {
        "selfplay": get_selfplay_orchestrator().get_status(),
        "pipeline": get_pipeline_orchestrator().get_status(),
        "task_lifecycle": get_task_lifecycle_coordinator().get_status(),
        "optimization": get_optimization_coordinator().get_status(),
        "metrics": get_metrics_orchestrator().get_status(),
        "resources": get_resource_coordinator().get_status(),
        "cache": get_cache_orchestrator().get_status(),
        "event_coordinator": get_event_coordinator_stats(),
    }
