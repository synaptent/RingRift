"""Cluster Health Checks.

Consolidates cluster_health_check.py, cluster_health_monitor.py, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from scripts.p2p.cluster_config import get_cluster_config
from .dashboard import get_cluster_status, NodeStatus


class HealthStatus(Enum):
    """Health check result status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: HealthStatus
    message: str
    details: str | None = None


@dataclass
class ClusterHealth:
    """Overall cluster health assessment."""
    status: HealthStatus
    checks: list[HealthCheckResult]
    summary: str


def check_node_health(node: NodeStatus) -> HealthCheckResult:
    """Check health of a single node."""
    config = get_cluster_config()
    thresholds = config.alerts

    if not node.online:
        return HealthCheckResult(
            status=HealthStatus.CRITICAL,
            message=f"{node.node_id} is offline",
        )

    issues = []

    # Check disk
    if node.disk_percent >= thresholds.disk_usage_critical:
        issues.append(f"disk critical ({node.disk_percent:.0f}%)")
    elif node.disk_percent >= thresholds.disk_usage_warn:
        issues.append(f"disk warning ({node.disk_percent:.0f}%)")

    # Check memory
    if node.memory_percent >= thresholds.memory_warn:
        issues.append(f"memory high ({node.memory_percent:.0f}%)")

    # Check GPU with active jobs
    if node.selfplay_jobs > 0 or node.training_jobs > 0:
        if node.gpu_percent <= thresholds.gpu_util_critical:
            issues.append(f"GPU idle with active jobs ({node.gpu_percent:.0f}%)")
        elif node.gpu_percent <= thresholds.gpu_util_low:
            issues.append(f"GPU low with active jobs ({node.gpu_percent:.0f}%)")

    if not issues:
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message=f"{node.node_id} healthy",
        )

    # Determine severity
    critical_keywords = ["critical", "offline", "GPU idle"]
    is_critical = any(kw in " ".join(issues) for kw in critical_keywords)

    return HealthCheckResult(
        status=HealthStatus.CRITICAL if is_critical else HealthStatus.DEGRADED,
        message=f"{node.node_id}: {', '.join(issues)}",
    )


def check_cluster_health(entry_point: str = "localhost:8770") -> ClusterHealth:
    """Check health of entire cluster.

    Args:
        entry_point: Host:port of any P2P orchestrator node.

    Returns:
        ClusterHealth with detailed assessment.
    """
    cluster = get_cluster_status(entry_point)

    if cluster.total_nodes == 0:
        return ClusterHealth(
            status=HealthStatus.UNKNOWN,
            checks=[],
            summary="Cannot reach cluster",
        )

    checks = []

    # Check each node
    for node in cluster.nodes:
        checks.append(check_node_health(node))

    # Check cluster-level issues
    if cluster.online_nodes == 0:
        checks.append(HealthCheckResult(
            status=HealthStatus.CRITICAL,
            message="No nodes online",
        ))
    elif cluster.online_nodes < cluster.total_nodes * 0.5:
        checks.append(HealthCheckResult(
            status=HealthStatus.CRITICAL,
            message=f"Less than 50% nodes online ({cluster.online_nodes}/{cluster.total_nodes})",
        ))

    if not cluster.leader:
        checks.append(HealthCheckResult(
            status=HealthStatus.DEGRADED,
            message="No cluster leader elected",
        ))

    if cluster.avg_gpu_util < 20 and cluster.total_selfplay_jobs > 0:
        checks.append(HealthCheckResult(
            status=HealthStatus.DEGRADED,
            message=f"Low average GPU utilization ({cluster.avg_gpu_util:.0f}%) with active jobs",
        ))

    # Determine overall status
    has_critical = any(c.status == HealthStatus.CRITICAL for c in checks)
    has_degraded = any(c.status == HealthStatus.DEGRADED for c in checks)

    if has_critical:
        overall_status = HealthStatus.CRITICAL
    elif has_degraded:
        overall_status = HealthStatus.DEGRADED
    else:
        overall_status = HealthStatus.HEALTHY

    # Generate summary
    critical_count = sum(1 for c in checks if c.status == HealthStatus.CRITICAL)
    degraded_count = sum(1 for c in checks if c.status == HealthStatus.DEGRADED)
    healthy_count = sum(1 for c in checks if c.status == HealthStatus.HEALTHY)

    summary = f"{healthy_count} healthy, {degraded_count} degraded, {critical_count} critical"

    return ClusterHealth(
        status=overall_status,
        checks=checks,
        summary=summary,
    )


def print_health_report(health: ClusterHealth) -> None:
    """Print health report in readable format."""
    status_icon = {
        HealthStatus.HEALTHY: "[OK]",
        HealthStatus.DEGRADED: "[WARN]",
        HealthStatus.CRITICAL: "[CRIT]",
        HealthStatus.UNKNOWN: "[???]",
    }

    print(f"\n{'='*60}")
    print(f"  CLUSTER HEALTH: {status_icon[health.status]} {health.status.value.upper()}")
    print(f"{'='*60}")
    print(f"  Summary: {health.summary}")
    print(f"{'='*60}\n")

    # Group by status
    for status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED, HealthStatus.HEALTHY]:
        checks = [c for c in health.checks if c.status == status]
        if checks:
            print(f"{status.value.upper()}:")
            for check in checks:
                print(f"  {status_icon[check.status]} {check.message}")
            print()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Cluster Health Check")
    parser.add_argument("--host", default="localhost", help="P2P orchestrator host")
    parser.add_argument("--port", type=int, default=8770, help="P2P orchestrator port")
    parser.add_argument("--exit-code", action="store_true", help="Exit with code based on health")
    args = parser.parse_args()

    entry = f"{args.host}:{args.port}"
    health = check_cluster_health(entry)

    print_health_report(health)

    if args.exit_code:
        if health.status == HealthStatus.CRITICAL:
            exit(2)
        elif health.status == HealthStatus.DEGRADED:
            exit(1)
        else:
            exit(0)


if __name__ == "__main__":
    main()
