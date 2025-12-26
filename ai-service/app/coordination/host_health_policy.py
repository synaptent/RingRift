#!/usr/bin/env python3
"""Host health policy for pre-spawn checks.

RENAMED from health_check.py for clarity - this module provides POLICY decisions
about whether hosts are healthy enough to receive jobs, distinct from
app/distributed/health_checks.py which performs the actual distributed probing.

This module provides fast, cached health checks for remote hosts before
spawning tasks. It integrates with the coordination system to prevent
spawning jobs on unhealthy or unreachable hosts.

Features:
- Fast SSH connectivity check with configurable timeout
- Result caching to avoid overloading hosts with checks
- Negative caching for failed hosts (temporary blacklist)
- Integration with coordination safeguards

Usage:
    from app.coordination.host_health_policy import (
        check_host_health,
        is_host_healthy,
        get_healthy_hosts,
        clear_health_cache,
    )

    # Quick check before spawning
    if is_host_healthy("gpu-server-1"):
        spawn_job("gpu-server-1")

    # Get all healthy hosts from a list
    healthy = get_healthy_hosts(["host-1", "host-2", "host-3"])
"""

from __future__ import annotations

import contextlib
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# Use centralized defaults (December 2025)
try:
    from app.config.coordination_defaults import HealthDefaults
    DEFAULT_SSH_TIMEOUT = HealthDefaults.SSH_TIMEOUT
    HEALTH_CACHE_TTL = HealthDefaults.HEALTHY_CACHE_TTL
    UNHEALTHY_CACHE_TTL = HealthDefaults.UNHEALTHY_CACHE_TTL
    MAX_CONCURRENT_CHECKS = HealthDefaults.MAX_CONCURRENT_CHECKS
except ImportError:
    # Fallback defaults
    DEFAULT_SSH_TIMEOUT = 5
    HEALTH_CACHE_TTL = 60
    UNHEALTHY_CACHE_TTL = 30
    MAX_CONCURRENT_CHECKS = 10


@dataclass
class HostHealthStatus:
    """Health status of a host.

    Note: This is host-specific health tracking.
    For generic health states (HEALTHY, DEGRADED, etc.), use app.core.health.HealthState.
    """

    host: str
    healthy: bool
    checked_at: float
    latency_ms: float | None = None
    error: str | None = None
    load_1m: float | None = None
    cpu_count: int | None = None

    @property
    def age_seconds(self) -> float:
        return time.time() - self.checked_at

    @property
    def is_stale(self) -> bool:
        ttl = HEALTH_CACHE_TTL if self.healthy else UNHEALTHY_CACHE_TTL
        return self.age_seconds > ttl

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "healthy": self.healthy,
            "checked_at": datetime.fromtimestamp(self.checked_at).isoformat(),
            "age_seconds": round(self.age_seconds, 1),
            "latency_ms": self.latency_ms,
            "error": self.error,
            "load_1m": self.load_1m,
            "cpu_count": self.cpu_count,
        }


# Global health cache
_health_cache: dict[str, HostHealthStatus] = {}
_cache_lock = threading.RLock()


def _get_ssh_target(host: str) -> tuple[str, str | None, int]:
    """Get SSH target, key path, and port for a host.

    Returns:
        Tuple of (ssh_target, key_path, port)
    """
    try:
        from app.distributed.hosts import load_remote_hosts

        hosts = load_remote_hosts()
        if host in hosts:
            host_config = hosts[host]
            return (
                host_config.ssh_target,
                host_config.ssh_key_path if host_config.ssh_key else None,
                host_config.ssh_port,
            )
    except ImportError:
        pass

    # Fallback: assume host is the SSH target directly
    return host, None, 22


def _quick_ssh_check(
    host: str,
    timeout: int = DEFAULT_SSH_TIMEOUT,
    check_load: bool = True,
) -> HostHealthStatus:
    """Perform a quick SSH health check on a host.

    Args:
        host: Host name (from distributed_hosts.yaml) or direct SSH target
        timeout: SSH timeout in seconds
        check_load: Whether to also check system load

    Returns:
        HealthStatus with check results
    """
    start_time = time.time()
    ssh_target, key_path, port = _get_ssh_target(host)

    # Build SSH command
    ssh_cmd = [
        "ssh",
        "-o", f"ConnectTimeout={timeout}",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
    ]

    if key_path:
        ssh_cmd.extend(["-i", key_path])
    if port != 22:
        ssh_cmd.extend(["-p", str(port)])

    # Command to run: echo for connectivity, optionally get load
    if check_load:
        remote_cmd = "echo OK && cat /proc/loadavg 2>/dev/null && nproc 2>/dev/null"
    else:
        remote_cmd = "echo OK"

    ssh_cmd.extend([ssh_target, remote_cmd])

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 2,  # Small buffer for process overhead
        )

        latency_ms = (time.time() - start_time) * 1000

        if result.returncode == 0 and "OK" in result.stdout:
            # Parse load if available
            load_1m = None
            cpu_count = None
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                try:
                    load_parts = lines[1].split()
                    load_1m = float(load_parts[0])
                except (IndexError, ValueError):
                    pass
            if len(lines) >= 3:
                with contextlib.suppress(ValueError):
                    cpu_count = int(lines[2])

            return HostHealthStatus(
                host=host,
                healthy=True,
                checked_at=time.time(),
                latency_ms=round(latency_ms, 1),
                load_1m=load_1m,
                cpu_count=cpu_count,
            )
        else:
            error_msg = result.stderr.strip() or f"Exit code {result.returncode}"
            return HostHealthStatus(
                host=host,
                healthy=False,
                checked_at=time.time(),
                latency_ms=round(latency_ms, 1),
                error=error_msg[:200],  # Truncate long errors
            )

    except subprocess.TimeoutExpired:
        return HostHealthStatus(
            host=host,
            healthy=False,
            checked_at=time.time(),
            error=f"SSH timeout after {timeout}s",
        )
    except Exception as e:
        return HostHealthStatus(
            host=host,
            healthy=False,
            checked_at=time.time(),
            error=str(e)[:200],
        )


def check_host_health(
    host: str,
    force_refresh: bool = False,
    timeout: int = DEFAULT_SSH_TIMEOUT,
) -> HostHealthStatus:
    """Check health of a host, using cache when available.

    Args:
        host: Host name to check
        force_refresh: If True, bypass cache and perform fresh check
        timeout: SSH timeout in seconds

    Returns:
        HealthStatus with current or cached status
    """
    # Check for localhost
    if host in ("localhost", "local", socket.gethostname()):
        return HostHealthStatus(
            host=host,
            healthy=True,
            checked_at=time.time(),
            latency_ms=0,
        )

    with _cache_lock:
        # Check cache
        if not force_refresh and host in _health_cache:
            cached = _health_cache[host]
            if not cached.is_stale:
                return cached

    # Perform fresh check
    status = _quick_ssh_check(host, timeout)

    with _cache_lock:
        _health_cache[host] = status

    return status


def is_host_healthy(host: str, force_refresh: bool = False) -> bool:
    """Quick check if a host is healthy.

    Args:
        host: Host name to check
        force_refresh: If True, bypass cache

    Returns:
        True if host is healthy
    """
    return check_host_health(host, force_refresh).healthy


def get_healthy_hosts(
    hosts: list[str],
    parallel: bool = True,
    force_refresh: bool = False,
) -> list[str]:
    """Get list of healthy hosts from a candidate list.

    Args:
        hosts: List of host names to check
        parallel: Whether to check hosts in parallel
        force_refresh: If True, bypass cache for all hosts

    Returns:
        List of healthy host names
    """
    if not hosts:
        return []

    if not parallel or len(hosts) <= 2:
        # Sequential check
        return [h for h in hosts if is_host_healthy(h, force_refresh)]

    # Parallel check with threading
    results: dict[str, bool] = {}
    threads: list[threading.Thread] = []
    results_lock = threading.RLock()

    def check_one(host: str):
        healthy = is_host_healthy(host, force_refresh)
        with results_lock:
            results[host] = healthy

    # Limit concurrent checks
    for i in range(0, len(hosts), MAX_CONCURRENT_CHECKS):
        batch = hosts[i:i + MAX_CONCURRENT_CHECKS]
        threads = []

        for host in batch:
            t = threading.Thread(target=check_one, args=(host,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join(timeout=DEFAULT_SSH_TIMEOUT + 5)

    return [h for h in hosts if results.get(h, False)]


def get_health_summary(hosts: list[str]) -> dict[str, Any]:
    """Get health summary for a list of hosts.

    Args:
        hosts: List of host names

    Returns:
        Summary dict with health statistics
    """
    statuses = [check_host_health(h) for h in hosts]

    healthy = [s for s in statuses if s.healthy]
    unhealthy = [s for s in statuses if not s.healthy]

    return {
        "total": len(hosts),
        "healthy": len(healthy),
        "unhealthy": len(unhealthy),
        "healthy_hosts": [s.host for s in healthy],
        "unhealthy_hosts": [{"host": s.host, "error": s.error} for s in unhealthy],
        "avg_latency_ms": (
            round(sum(s.latency_ms for s in healthy if s.latency_ms) / len(healthy), 1)
            if healthy
            else None
        ),
    }


def clear_health_cache(host: str | None = None) -> int:
    """Clear health cache.

    Args:
        host: Specific host to clear, or None to clear all

    Returns:
        Number of entries cleared
    """
    global _health_cache
    with _cache_lock:
        if host:
            if host in _health_cache:
                del _health_cache[host]
                return 1
            return 0
        else:
            count = len(_health_cache)
            _health_cache = {}
            return count


def mark_host_unhealthy(host: str, error: str = "Manually marked unhealthy") -> None:
    """Manually mark a host as unhealthy (e.g., after task failure).

    This is useful when a host appears healthy via SSH but fails to run tasks.
    """
    with _cache_lock:
        _health_cache[host] = HostHealthStatus(
            host=host,
            healthy=False,
            checked_at=time.time(),
            error=error,
        )


def get_cache_status() -> dict[str, Any]:
    """Get current cache status."""
    with _cache_lock:
        entries = []
        for host, status in _health_cache.items():
            entries.append({
                "host": host,
                "healthy": status.healthy,
                "age_seconds": round(status.age_seconds, 1),
                "stale": status.is_stale,
            })

        return {
            "total_entries": len(_health_cache),
            "entries": entries,
            "healthy_ttl": HEALTH_CACHE_TTL,
            "unhealthy_ttl": UNHEALTHY_CACHE_TTL,
        }


# Integration with pre-spawn checks


def pre_spawn_check(
    host: str,
    task_type: str = "selfplay",
    check_load: bool = True,
    max_load_per_cpu: float = 0.8,
) -> tuple[bool, str]:
    """Combined pre-spawn check including health and load.

    This should be called before spawning any task on a remote host.

    Args:
        host: Host to check
        task_type: Type of task being spawned (for logging)
        check_load: Whether to check system load
        max_load_per_cpu: Maximum load per CPU to allow spawning

    Returns:
        Tuple of (can_spawn: bool, reason: str)
    """
    status = check_host_health(host)

    if not status.healthy:
        return False, f"Host unhealthy: {status.error}"

    if check_load and status.load_1m is not None and status.cpu_count:
        load_per_cpu = status.load_1m / status.cpu_count
        if load_per_cpu > max_load_per_cpu:
            return False, f"Host load too high: {status.load_1m:.1f}/{status.cpu_count} CPUs ({load_per_cpu:.2f}/CPU)"

    return True, f"OK (latency: {status.latency_ms:.0f}ms)" if status.latency_ms else "OK"


# =============================================================================
# Cluster-wide Health Gates
# =============================================================================

# Default minimum healthy hosts for expensive operations
# Use centralized defaults (December 2025)
try:
    DEFAULT_MIN_HEALTHY_HOSTS = HealthDefaults.MIN_HEALTHY_HOSTS
    CLUSTER_HEALTH_CACHE_TTL = HealthDefaults.CLUSTER_HEALTH_CACHE_TTL
except NameError:
    # Fallback if HealthDefaults not imported
    DEFAULT_MIN_HEALTHY_HOSTS = 2
    CLUSTER_HEALTH_CACHE_TTL = 120

# Cache for cluster health status
_cluster_health_cache: dict[str, Any] = {}
_cluster_health_lock = threading.RLock()


def check_cluster_health(
    hosts: list[str] | None = None,
    min_healthy: int = DEFAULT_MIN_HEALTHY_HOSTS,
    min_healthy_ratio: float = 0.5,
    force_refresh: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Check if cluster has enough healthy hosts for expensive operations.

    Use this before starting training, large evaluations, or critical syncs
    to avoid wasting resources when the cluster is degraded.

    Args:
        hosts: List of hosts to check. If None, loads from distributed_hosts.yaml
        min_healthy: Minimum absolute number of healthy hosts required
        min_healthy_ratio: Minimum ratio of healthy hosts (e.g., 0.5 = 50%)
        force_refresh: Bypass cache and check all hosts

    Returns:
        Tuple of (cluster_healthy: bool, details: Dict) with health summary
    """

    # Load hosts if not provided
    if hosts is None:
        try:
            from app.distributed.hosts import load_remote_hosts
            hosts = list(load_remote_hosts().keys())
        except ImportError:
            # No distributed hosts configured, cluster check passes
            return True, {"status": "ok", "reason": "no distributed hosts configured"}

    if not hosts:
        return True, {"status": "ok", "reason": "no hosts to check"}

    # Check cache (unless force refresh)
    cache_key = f"{','.join(sorted(hosts))}:{min_healthy}:{min_healthy_ratio}"
    with _cluster_health_lock:
        if not force_refresh and cache_key in _cluster_health_cache:
            cached = _cluster_health_cache[cache_key]
            if time.time() - cached["checked_at"] < CLUSTER_HEALTH_CACHE_TTL:
                return cached["healthy"], cached

    # Perform health checks in parallel
    healthy_hosts = get_healthy_hosts(hosts, force_refresh=force_refresh, parallel=True)

    total = len(hosts)
    healthy = len(healthy_hosts)
    unhealthy = total - healthy
    ratio = healthy / total if total > 0 else 0

    # Determine cluster health
    is_healthy = healthy >= min_healthy and ratio >= min_healthy_ratio

    details = {
        "healthy": is_healthy,
        "status": "healthy" if is_healthy else "degraded",
        "checked_at": time.time(),
        "total_hosts": total,
        "healthy_hosts": healthy,
        "unhealthy_hosts": unhealthy,
        "healthy_ratio": round(ratio, 2),
        "healthy_host_names": healthy_hosts,
        "unhealthy_host_names": [h for h in hosts if h not in healthy_hosts],
        "min_required": min_healthy,
        "min_ratio_required": min_healthy_ratio,
    }

    if not is_healthy:
        if healthy < min_healthy:
            details["reason"] = f"Only {healthy}/{min_healthy} required hosts healthy"
        else:
            details["reason"] = f"Only {ratio:.0%} of hosts healthy (need {min_healthy_ratio:.0%})"
    else:
        details["reason"] = f"{healthy}/{total} hosts healthy ({ratio:.0%})"

    # Cache result
    with _cluster_health_lock:
        _cluster_health_cache[cache_key] = details

    return is_healthy, details


def is_cluster_healthy(
    min_healthy: int = DEFAULT_MIN_HEALTHY_HOSTS,
    min_healthy_ratio: float = 0.5,
) -> bool:
    """Quick check if cluster is healthy enough for expensive operations.

    Args:
        min_healthy: Minimum number of healthy hosts
        min_healthy_ratio: Minimum ratio of healthy hosts

    Returns:
        True if cluster is healthy, False otherwise
    """
    healthy, _ = check_cluster_health(
        min_healthy=min_healthy,
        min_healthy_ratio=min_healthy_ratio,
    )
    return healthy


def gate_on_cluster_health(
    operation_name: str,
    min_healthy: int = DEFAULT_MIN_HEALTHY_HOSTS,
    min_healthy_ratio: float = 0.5,
) -> tuple[bool, str]:
    """Health gate for expensive operations.

    Use this at the start of expensive operations to fail fast if the
    cluster is unhealthy.

    Args:
        operation_name: Name of the operation (for logging)
        min_healthy: Minimum healthy hosts required
        min_healthy_ratio: Minimum healthy ratio required

    Returns:
        Tuple of (can_proceed: bool, message: str)

    Example:
        can_proceed, msg = gate_on_cluster_health("training")
        if not can_proceed:
            print(f"[Training] Skipping: {msg}")
            return
    """
    healthy, details = check_cluster_health(
        min_healthy=min_healthy,
        min_healthy_ratio=min_healthy_ratio,
    )

    if healthy:
        return True, f"Cluster healthy: {details['reason']}"
    else:
        return False, f"Cluster degraded: {details['reason']}"


# Command-line interface

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Host health check utilities")
    parser.add_argument("--check", type=str, nargs="+", help="Check specific host(s)")
    parser.add_argument("--all", action="store_true", help="Check all configured hosts")
    parser.add_argument("--cluster", action="store_true", help="Check cluster health gate")
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    parser.add_argument("--cache", action="store_true", help="Show cache status")
    parser.add_argument("--clear", action="store_true", help="Clear health cache")
    parser.add_argument("--force", action="store_true", help="Force refresh (bypass cache)")
    parser.add_argument("--min-healthy", type=int, default=2, help="Min healthy hosts for cluster check")
    parser.add_argument("--min-ratio", type=float, default=0.5, help="Min healthy ratio for cluster check")
    args = parser.parse_args()

    if args.clear:
        count = clear_health_cache()
        print(f"Cleared {count} cache entries")

    elif args.cache:
        print(json.dumps(get_cache_status(), indent=2))

    elif args.cluster:
        # Cluster health gate check
        healthy, details = check_cluster_health(
            min_healthy=args.min_healthy,
            min_healthy_ratio=args.min_ratio,
            force_refresh=args.force,
        )
        status_icon = "✓" if healthy else "✗"
        print(f"{status_icon} Cluster {details['status']}: {details.get('reason', 'unknown')}")
        print(f"  Hosts: {details.get('healthy_hosts', 0)}/{details.get('total_hosts', 0)} healthy")
        if details.get('unhealthy_host_names'):
            print(f"  Unhealthy: {', '.join(details['unhealthy_host_names'])}")
        # Exit with appropriate code for scripting
        sys.exit(0 if healthy else 1)

    elif args.all:
        try:
            from app.distributed.hosts import load_remote_hosts

            hosts = list(load_remote_hosts().keys())
            if args.summary:
                print(json.dumps(get_health_summary(hosts), indent=2))
            else:
                healthy = get_healthy_hosts(hosts, force_refresh=args.force)
                print(f"Healthy hosts ({len(healthy)}/{len(hosts)}):")
                for h in healthy:
                    status = check_host_health(h)
                    print(f"  {h}: {status.latency_ms:.0f}ms" if status.latency_ms else f"  {h}")
        except ImportError:
            print("No distributed_hosts.yaml found")

    elif args.check:
        for host in args.check:
            status = check_host_health(host, force_refresh=args.force)
            print(json.dumps(status.to_dict(), indent=2))

    else:
        parser.print_help()


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Data classes
    "HostHealthStatus",
    "check_cluster_health",
    # Functions
    "check_host_health",
    "clear_health_cache",
    "gate_on_cluster_health",
    "get_cache_status",
    "get_health_summary",
    "get_healthy_hosts",
    "is_cluster_healthy",
    "is_host_healthy",
    "mark_host_unhealthy",
    "pre_spawn_check",
]
