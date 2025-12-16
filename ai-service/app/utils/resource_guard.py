"""Unified Resource Guard - Pre-operation resource checks.

Provides simple functions for scripts to check resource availability
before performing operations. Enforces consistent 80% utilization limits.

Usage:
    from app.utils.resource_guard import (
        check_disk_space,
        check_memory,
        check_cpu,
        check_gpu_memory,
        can_proceed,
        wait_for_resources,
    )

    # Before writing files
    if not check_disk_space(required_gb=2.0):
        logger.warning("Insufficient disk space, skipping write")
        return

    # Before starting computation
    if not can_proceed():
        wait_for_resources(timeout=300)

    # In long-running loops
    for i in range(num_games):
        if i % 50 == 0 and not check_memory():
            logger.warning("Memory pressure, stopping early")
            break
        play_game()
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================
# Resource Limits - 80% max utilization
# ============================================

@dataclass(frozen=True)
class ResourceLimits:
    """Unified resource limits - 80% max utilization."""
    # Disk is tighter at 70% because cleanup takes time
    DISK_MAX_PERCENT: float = 70.0
    DISK_WARN_PERCENT: float = 65.0

    # CPU/GPU/Memory at 80% hard limit
    CPU_MAX_PERCENT: float = 80.0
    CPU_WARN_PERCENT: float = 70.0

    GPU_MAX_PERCENT: float = 80.0
    GPU_WARN_PERCENT: float = 70.0

    MEMORY_MAX_PERCENT: float = 80.0
    MEMORY_WARN_PERCENT: float = 70.0

    # Load average limit
    LOAD_MAX_FACTOR: float = 1.5  # load > cpus * this = overloaded


LIMITS = ResourceLimits()


# ============================================
# Disk Space Checks
# ============================================

def get_disk_usage(path: Optional[str] = None) -> Tuple[float, float, float]:
    """Get disk usage for a path.

    Args:
        path: Path to check (defaults to ai-service data directory)

    Returns:
        Tuple of (used_percent, available_gb, total_gb)
    """
    if path is None:
        path = str(Path(__file__).parent.parent.parent / "data")

    try:
        usage = shutil.disk_usage(path)
        total_gb = usage.total / (1024 ** 3)
        available_gb = usage.free / (1024 ** 3)
        used_percent = ((usage.total - usage.free) / usage.total) * 100
        return used_percent, available_gb, total_gb
    except Exception as e:
        logger.warning(f"Could not check disk usage: {e}")
        return 0.0, 999.0, 999.0  # Assume OK on error


def check_disk_space(
    required_gb: float = 2.0,
    path: Optional[str] = None,
    log_warning: bool = True,
) -> bool:
    """Check if sufficient disk space is available.

    Args:
        required_gb: Minimum free space required in GB
        path: Path to check
        log_warning: Whether to log a warning on failure

    Returns:
        True if sufficient space available
    """
    used_percent, available_gb, _ = get_disk_usage(path)

    # Check both percentage limit and absolute requirement
    if used_percent >= LIMITS.DISK_MAX_PERCENT:
        if log_warning:
            logger.warning(
                f"Disk usage {used_percent:.1f}% exceeds limit {LIMITS.DISK_MAX_PERCENT}%"
            )
        return False

    if available_gb < required_gb:
        if log_warning:
            logger.warning(
                f"Disk space {available_gb:.1f}GB below required {required_gb:.1f}GB"
            )
        return False

    return True


def check_disk_for_write(
    estimated_size_mb: float,
    path: Optional[str] = None,
) -> bool:
    """Check if disk has space for an estimated write operation.

    Args:
        estimated_size_mb: Estimated size of data to write in MB
        path: Path to check

    Returns:
        True if write is safe to proceed
    """
    required_gb = (estimated_size_mb / 1024) + 1.0  # Add 1GB safety margin
    return check_disk_space(required_gb=required_gb, path=path)


# ============================================
# Memory Checks
# ============================================

def get_memory_usage() -> Tuple[float, float, float]:
    """Get system memory usage.

    Returns:
        Tuple of (used_percent, available_gb, total_gb)
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        available_gb = mem.available / (1024 ** 3)
        used_percent = mem.percent
        return used_percent, available_gb, total_gb
    except ImportError:
        logger.debug("psutil not available for memory check")
        return 0.0, 999.0, 999.0
    except Exception as e:
        logger.warning(f"Could not check memory: {e}")
        return 0.0, 999.0, 999.0


def check_memory(
    required_gb: float = 1.0,
    log_warning: bool = True,
) -> bool:
    """Check if sufficient memory is available.

    Args:
        required_gb: Minimum free memory required in GB
        log_warning: Whether to log a warning on failure

    Returns:
        True if sufficient memory available
    """
    used_percent, available_gb, _ = get_memory_usage()

    if used_percent >= LIMITS.MEMORY_MAX_PERCENT:
        if log_warning:
            logger.warning(
                f"Memory usage {used_percent:.1f}% exceeds limit {LIMITS.MEMORY_MAX_PERCENT}%"
            )
        return False

    if available_gb < required_gb:
        if log_warning:
            logger.warning(
                f"Memory {available_gb:.1f}GB below required {required_gb:.1f}GB"
            )
        return False

    return True


# ============================================
# CPU Checks
# ============================================

def get_cpu_usage() -> Tuple[float, float, int]:
    """Get CPU usage and load average.

    Returns:
        Tuple of (cpu_percent, load_per_cpu, cpu_count)
    """
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count() or 1
        load_1min = os.getloadavg()[0]
        load_per_cpu = load_1min / cpu_count
        return cpu_percent, load_per_cpu, cpu_count
    except (ImportError, AttributeError):
        return 0.0, 0.0, 1
    except Exception as e:
        logger.warning(f"Could not check CPU: {e}")
        return 0.0, 0.0, 1


def check_cpu(log_warning: bool = True) -> bool:
    """Check if CPU usage is within limits.

    Returns:
        True if CPU usage is acceptable
    """
    cpu_percent, load_per_cpu, _ = get_cpu_usage()

    if cpu_percent >= LIMITS.CPU_MAX_PERCENT:
        if log_warning:
            logger.warning(
                f"CPU usage {cpu_percent:.1f}% exceeds limit {LIMITS.CPU_MAX_PERCENT}%"
            )
        return False

    if load_per_cpu >= LIMITS.LOAD_MAX_FACTOR:
        if log_warning:
            logger.warning(
                f"Load average {load_per_cpu:.2f}x exceeds limit {LIMITS.LOAD_MAX_FACTOR}x"
            )
        return False

    return True


# ============================================
# GPU Memory Checks
# ============================================

def get_gpu_memory_usage(device_id: int = 0) -> Tuple[float, float, float]:
    """Get GPU memory usage.

    Args:
        device_id: CUDA device ID

    Returns:
        Tuple of (used_percent, available_gb, total_gb)
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0, 0.0, 0.0

        torch.cuda.set_device(device_id)
        total = torch.cuda.get_device_properties(device_id).total_memory
        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)

        # Use reserved memory as "used" since it's not available
        used = max(allocated, reserved)
        available = total - used

        total_gb = total / (1024 ** 3)
        available_gb = available / (1024 ** 3)
        used_percent = (used / total) * 100 if total > 0 else 0

        return used_percent, available_gb, total_gb
    except ImportError:
        return 0.0, 0.0, 0.0
    except Exception as e:
        logger.debug(f"Could not check GPU memory: {e}")
        return 0.0, 0.0, 0.0


def check_gpu_memory(
    required_gb: float = 1.0,
    device_id: int = 0,
    log_warning: bool = True,
) -> bool:
    """Check if sufficient GPU memory is available.

    Args:
        required_gb: Minimum free GPU memory in GB
        device_id: CUDA device ID
        log_warning: Whether to log a warning on failure

    Returns:
        True if sufficient GPU memory available
    """
    used_percent, available_gb, total_gb = get_gpu_memory_usage(device_id)

    if total_gb == 0:
        return True  # No GPU or not using CUDA

    if used_percent >= LIMITS.GPU_MAX_PERCENT:
        if log_warning:
            logger.warning(
                f"GPU memory {used_percent:.1f}% exceeds limit {LIMITS.GPU_MAX_PERCENT}%"
            )
        return False

    if available_gb < required_gb:
        if log_warning:
            logger.warning(
                f"GPU memory {available_gb:.1f}GB below required {required_gb:.1f}GB"
            )
        return False

    return True


def clear_gpu_memory():
    """Clear GPU memory cache."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
    except ImportError:
        pass


# ============================================
# Combined Checks
# ============================================

def can_proceed(
    check_disk: bool = True,
    check_mem: bool = True,
    check_cpu_load: bool = True,
    check_gpu: bool = False,
    disk_required_gb: float = 2.0,
    mem_required_gb: float = 1.0,
    gpu_required_gb: float = 1.0,
) -> bool:
    """Check if all specified resources are within limits.

    Args:
        check_disk: Whether to check disk space
        check_mem: Whether to check memory
        check_cpu_load: Whether to check CPU usage
        check_gpu: Whether to check GPU memory
        disk_required_gb: Minimum disk space
        mem_required_gb: Minimum memory
        gpu_required_gb: Minimum GPU memory

    Returns:
        True if all checked resources are within limits
    """
    if check_disk and not check_disk_space(disk_required_gb, log_warning=False):
        return False

    if check_mem and not check_memory(mem_required_gb, log_warning=False):
        return False

    if check_cpu_load and not check_cpu(log_warning=False):
        return False

    if check_gpu and not check_gpu_memory(gpu_required_gb, log_warning=False):
        return False

    return True


def wait_for_resources(
    timeout: float = 300.0,
    check_interval: float = 10.0,
    disk_required_gb: float = 2.0,
    mem_required_gb: float = 1.0,
) -> bool:
    """Wait for resources to become available.

    Args:
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks in seconds
        disk_required_gb: Minimum disk space
        mem_required_gb: Minimum memory

    Returns:
        True if resources became available, False if timeout
    """
    start = time.time()

    while time.time() - start < timeout:
        if can_proceed(
            disk_required_gb=disk_required_gb,
            mem_required_gb=mem_required_gb,
            check_gpu=False,
        ):
            return True

        elapsed = time.time() - start
        logger.info(
            f"Waiting for resources... ({elapsed:.0f}s/{timeout:.0f}s)"
        )
        time.sleep(check_interval)

    logger.warning(f"Resource wait timed out after {timeout}s")
    return False


# ============================================
# Resource Status Report
# ============================================

def get_resource_status() -> dict:
    """Get current resource status as a dictionary.

    Returns:
        Dict with resource usage and status
    """
    disk_pct, disk_avail, disk_total = get_disk_usage()
    mem_pct, mem_avail, mem_total = get_memory_usage()
    cpu_pct, load_per_cpu, cpu_count = get_cpu_usage()
    gpu_pct, gpu_avail, gpu_total = get_gpu_memory_usage()

    return {
        "disk": {
            "used_percent": round(disk_pct, 1),
            "available_gb": round(disk_avail, 1),
            "total_gb": round(disk_total, 1),
            "ok": disk_pct < LIMITS.DISK_MAX_PERCENT,
            "limit_percent": LIMITS.DISK_MAX_PERCENT,
        },
        "memory": {
            "used_percent": round(mem_pct, 1),
            "available_gb": round(mem_avail, 1),
            "total_gb": round(mem_total, 1),
            "ok": mem_pct < LIMITS.MEMORY_MAX_PERCENT,
            "limit_percent": LIMITS.MEMORY_MAX_PERCENT,
        },
        "cpu": {
            "used_percent": round(cpu_pct, 1),
            "load_per_cpu": round(load_per_cpu, 2),
            "cpu_count": cpu_count,
            "ok": cpu_pct < LIMITS.CPU_MAX_PERCENT and load_per_cpu < LIMITS.LOAD_MAX_FACTOR,
            "limit_percent": LIMITS.CPU_MAX_PERCENT,
        },
        "gpu": {
            "used_percent": round(gpu_pct, 1),
            "available_gb": round(gpu_avail, 1),
            "total_gb": round(gpu_total, 1),
            "ok": gpu_pct < LIMITS.GPU_MAX_PERCENT or gpu_total == 0,
            "limit_percent": LIMITS.GPU_MAX_PERCENT,
        },
        "can_proceed": can_proceed(check_gpu=False),
    }


def print_resource_status():
    """Print formatted resource status."""
    status = get_resource_status()

    print("=" * 50)
    print("RESOURCE STATUS (80% max utilization)")
    print("=" * 50)

    for name, info in status.items():
        if name == "can_proceed":
            continue
        symbol = "✓" if info["ok"] else "✗"
        print(f"\n{name.upper()}:")
        print(f"  {symbol} Used: {info['used_percent']:.1f}% (limit: {info['limit_percent']}%)")
        if "available_gb" in info:
            print(f"    Available: {info['available_gb']:.1f} GB")

    overall = "✓ OK to proceed" if status["can_proceed"] else "✗ Resource limits exceeded"
    print(f"\n{overall}")
    print("=" * 50)


# ============================================
# Context Manager for Resource-Safe Operations
# ============================================

class ResourceGuard:
    """Context manager for resource-safe operations.

    Usage:
        with ResourceGuard(disk_required_gb=5.0, mem_required_gb=2.0) as guard:
            if not guard.ok:
                return  # Resources not available
            # ... do work ...
    """

    def __init__(
        self,
        disk_required_gb: float = 2.0,
        mem_required_gb: float = 1.0,
        gpu_required_gb: float = 0.0,
        wait_timeout: float = 0.0,
    ):
        self.disk_required_gb = disk_required_gb
        self.mem_required_gb = mem_required_gb
        self.gpu_required_gb = gpu_required_gb
        self.wait_timeout = wait_timeout
        self.ok = False

    def __enter__(self):
        check_gpu = self.gpu_required_gb > 0

        if self.wait_timeout > 0:
            self.ok = wait_for_resources(
                timeout=self.wait_timeout,
                disk_required_gb=self.disk_required_gb,
                mem_required_gb=self.mem_required_gb,
            )
        else:
            self.ok = can_proceed(
                disk_required_gb=self.disk_required_gb,
                mem_required_gb=self.mem_required_gb,
                check_gpu=check_gpu,
                gpu_required_gb=self.gpu_required_gb,
            )

        if not self.ok:
            status = get_resource_status()
            logger.warning(f"Resource limits exceeded: {status}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clear GPU memory on exit if we used it
        if self.gpu_required_gb > 0:
            clear_gpu_memory()
        return False


# ============================================
# Convenience for Scripts
# ============================================

def require_resources(
    disk_gb: float = 2.0,
    mem_gb: float = 1.0,
    gpu_gb: float = 0.0,
    wait_timeout: float = 60.0,
) -> bool:
    """Require resources before proceeding, waiting if necessary.

    Use at the start of scripts to ensure resources are available.

    Args:
        disk_gb: Required disk space in GB
        mem_gb: Required memory in GB
        gpu_gb: Required GPU memory in GB (0 to skip check)
        wait_timeout: Time to wait for resources (0 for no wait)

    Returns:
        True if resources available, False otherwise
    """
    with ResourceGuard(
        disk_required_gb=disk_gb,
        mem_required_gb=mem_gb,
        gpu_required_gb=gpu_gb,
        wait_timeout=wait_timeout,
    ) as guard:
        return guard.ok


if __name__ == "__main__":
    # Print status when run directly
    print_resource_status()
