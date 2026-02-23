"""P2P Orchestrator Resource Utilities.

This module contains resource checking functions for the P2P orchestrator.
All resource checks use the unified 80% max utilization thresholds.
Extracted from p2p_orchestrator.py for better modularity.
"""

from __future__ import annotations

import os

from .constants import MAX_DISK_USAGE_PERCENT

# Unified resource checking utilities (80% max utilization)
try:
    from app.utils.resource_guard import (
        check_cpu as unified_check_cpu,
        check_disk_space as unified_check_disk,
        check_memory as unified_check_memory,
        get_cpu_usage,
        get_disk_usage,
        get_memory_usage,
    )
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    unified_check_disk = None
    unified_check_memory = None
    unified_check_cpu = None
    get_disk_usage = None
    get_memory_usage = None
    get_cpu_usage = None


def get_disk_usage_percent(path: str | None = None) -> float:
    """Get disk usage percentage for the given path.

    Uses unified resource_guard utilities when available for consistent
    80% max utilization enforcement across the codebase.

    Returns:
        Disk usage as a percentage (0-100), or 100.0 on error.
    """
    if HAS_RESOURCE_GUARD and get_disk_usage is not None:
        try:
            used_pct, _, _ = get_disk_usage(path)
            return used_pct
        except (RuntimeError, OSError, ValueError):
            pass
    # Fallback to original implementation
    check_path = path or os.path.dirname(os.path.abspath(__file__))
    try:
        stat = os.statvfs(check_path)
        total = stat.f_blocks * stat.f_frsize
        free = stat.f_bavail * stat.f_frsize
        if total <= 0:
            return 100.0
        used = total - free
        return (used / total) * 100.0
    except (OSError, ValueError):
        return 100.0  # Assume full on error to be safe


def check_disk_has_capacity(threshold: float | None = None) -> tuple[bool, float]:
    """Check if disk has capacity below threshold for data sync.

    Uses unified resource_guard utilities when available for consistent
    85% max utilization enforcement for disk.

    Args:
        threshold: Max disk usage percentage (defaults to MAX_DISK_USAGE_PERCENT)

    Returns:
        Tuple of (has_capacity: bool, current_percent: float)
    """
    threshold = threshold if threshold is not None else MAX_DISK_USAGE_PERCENT
    current = get_disk_usage_percent()
    return (current < threshold, current)


def check_all_resources() -> tuple[bool, str]:
    """Check if all resources (CPU, memory, disk) are within limits.

    Uses unified 80% max utilization thresholds from resource_guard.

    Returns:
        Tuple of (can_proceed: bool, reason: str)
    """
    if not HAS_RESOURCE_GUARD:
        # Fallback: only check disk
        has_disk, disk_pct = check_disk_has_capacity()
        if not has_disk:
            return False, f"Disk at {disk_pct:.1f}%"
        return True, "OK"

    reasons = []

    # Check disk (85% limit)
    if not unified_check_disk(log_warning=False):
        disk_pct, _, _ = get_disk_usage()
        reasons.append(f"Disk {disk_pct:.1f}%")

    # Check memory (80% limit)
    if not unified_check_memory(log_warning=False):
        mem_pct, _, _ = get_memory_usage()
        reasons.append(f"Memory {mem_pct:.1f}%")

    # Check CPU (80% limit)
    if not unified_check_cpu(log_warning=False):
        cpu_pct, _, _ = get_cpu_usage()
        reasons.append(f"CPU {cpu_pct:.1f}%")

    if reasons:
        return False, ", ".join(reasons)
    return True, "OK"
