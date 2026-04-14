"""Status and health-report helpers for coordination package consumers."""

from __future__ import annotations

import time as _time

from app.coordination.cache_coordination_orchestrator import get_cache_orchestrator
from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator
from app.coordination.event_router import get_coordinator_stats as get_event_coordinator_stats
from app.coordination.metrics_analysis_orchestrator import get_metrics_orchestrator
from app.coordination.optimization_coordinator import get_optimization_coordinator
from app.coordination.resource_monitoring_coordinator import get_resource_coordinator
from app.coordination.selfplay_orchestrator import get_selfplay_orchestrator
from app.coordination.task_lifecycle_coordinator import get_task_lifecycle_coordinator

__all__ = ["get_all_coordinator_status", "get_system_health"]


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


def _get_health_score(name: str, status: dict, issues: list[str]) -> float:
    """Calculate a health score for one coordinator status payload."""

    score = 1.0

    if not status.get("subscribed", True):
        score -= 0.3
        issues.append(f"{name}: not subscribed to events")

    if status.get("paused", False):
        score -= 0.2
        issues.append(f"{name}: paused ({status.get('pause_reason', 'unknown')})")

    if status.get("resource_constraints"):
        for constraint_type, constraint in status.get("resource_constraints", {}).items():
            if isinstance(constraint, dict) and constraint.get("severity") == "critical":
                score -= 0.2
                issues.append(f"{name}: critical {constraint_type} constraint")

    if status.get("backpressure_active"):
        score -= 0.1
        issues.append(f"{name}: backpressure active")

    if status.get("plateaus"):
        score -= 0.1 * min(len(status["plateaus"]), 3)
        for metric in status["plateaus"][:3]:
            issues.append(f"{name}: plateau detected in {metric}")

    if status.get("regressions"):
        score -= 0.2 * min(len(status["regressions"]), 2)
        for metric in status["regressions"][:2]:
            issues.append(f"{name}: regression detected in {metric}")

    if status.get("orphaned", 0) > 0:
        orphan_count = status["orphaned"]
        score -= 0.1 * min(orphan_count, 5)
        issues.append(f"{name}: {orphan_count} orphaned tasks")

    if status.get("failed_tasks", 0) > 10:
        score -= 0.1
        issues.append(f"{name}: high failure count ({status['failed_tasks']})")

    return max(0.0, score)


def get_system_health() -> dict:
    """Get aggregated health from all coordinators."""

    issues: list[str] = []
    coordinator_health = {}
    total_score = 0.0
    coordinator_count = 0

    coordinators = [
        ("selfplay", get_selfplay_orchestrator),
        ("pipeline", get_pipeline_orchestrator),
        ("task_lifecycle", get_task_lifecycle_coordinator),
        ("optimization", get_optimization_coordinator),
        ("metrics", get_metrics_orchestrator),
        ("resources", get_resource_coordinator),
        ("cache", get_cache_orchestrator),
    ]

    for name, getter in coordinators:
        try:
            status = getter().get_status()
            coordinator_health[name] = _get_health_score(name, status, issues)
            coordinator_count += 1
            total_score += coordinator_health[name]
        except Exception as exc:
            coordinator_health[name] = 0.0
            issues.append(f"{name}: failed to get status ({exc})")

    overall_health = total_score / coordinator_count if coordinator_count > 0 else 0.0

    if overall_health >= 0.9:
        status_str = "healthy"
    elif overall_health >= 0.7:
        status_str = "degraded"
    else:
        status_str = "unhealthy"

    handler_health = {}
    try:
        from app.coordination.handler_resilience import get_all_handler_metrics

        all_metrics = get_all_handler_metrics()
        total_invocations = sum(metric.invocation_count for metric in all_metrics.values())
        total_failures = sum(metric.failure_count for metric in all_metrics.values())
        total_timeouts = sum(metric.timeout_count for metric in all_metrics.values())

        handler_health = {
            "total_handlers": len(all_metrics),
            "total_invocations": total_invocations,
            "total_failures": total_failures,
            "total_timeouts": total_timeouts,
            "success_rate": (
                (total_invocations - total_failures - total_timeouts) / total_invocations
                if total_invocations > 0
                else 1.0
            ),
            "unhealthy_handlers": [
                name for name, metric in all_metrics.items() if metric.consecutive_failures >= 3
            ],
        }

        if handler_health["unhealthy_handlers"]:
            for handler in handler_health["unhealthy_handlers"]:
                issues.append(f"handler: {handler} has consecutive failures")
    except (AttributeError, ImportError, KeyError):
        pass

    return {
        "overall_health": round(overall_health, 3),
        "status": status_str,
        "coordinators": coordinator_health,
        "issues": issues[:20],
        "handler_health": handler_health,
        "timestamp": _time.time(),
    }
