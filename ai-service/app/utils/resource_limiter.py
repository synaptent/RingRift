"""Resource Limiter - Prevents overloading CPU, GPU, RAM, and disk.

This module provides utilities to check resource utilization and back off
when resources exceed 80% usage. All long-running processes should use
these utilities to prevent system overload.

Usage:
    from app.utils.resource_limiter import ResourceLimiter, check_resources

    # Quick check before starting work
    if not check_resources():
        logger.warning("Resources over 80%, backing off")
        time.sleep(60)

    # Or use the limiter context manager
    limiter = ResourceLimiter()
    async with limiter.acquire("training"):
        # Do work - automatically backs off if resources high
        train_model()
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default thresholds (80%)
DEFAULT_CPU_THRESHOLD = 80.0
DEFAULT_GPU_THRESHOLD = 80.0
DEFAULT_RAM_THRESHOLD = 80.0
DEFAULT_DISK_THRESHOLD = 80.0

# Backoff settings
MIN_BACKOFF_SECONDS = 5
MAX_BACKOFF_SECONDS = 300
BACKOFF_MULTIPLIER = 1.5


@dataclass
class ResourceStatus:
    """Current resource utilization status."""
    cpu_percent: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    ram_percent: float = 0.0
    disk_percent: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def is_overloaded(
        self,
        cpu_threshold: float = DEFAULT_CPU_THRESHOLD,
        gpu_threshold: float = DEFAULT_GPU_THRESHOLD,
        ram_threshold: float = DEFAULT_RAM_THRESHOLD,
        disk_threshold: float = DEFAULT_DISK_THRESHOLD,
    ) -> Tuple[bool, List[str]]:
        """Check if any resource exceeds threshold.

        Returns:
            Tuple of (is_overloaded, list of overloaded resources)
        """
        overloaded = []

        if self.cpu_percent > cpu_threshold:
            overloaded.append(f"CPU: {self.cpu_percent:.1f}%")
        if self.gpu_percent > gpu_threshold:
            overloaded.append(f"GPU: {self.gpu_percent:.1f}%")
        if self.gpu_memory_percent > gpu_threshold:
            overloaded.append(f"GPU Memory: {self.gpu_memory_percent:.1f}%")
        if self.ram_percent > ram_threshold:
            overloaded.append(f"RAM: {self.ram_percent:.1f}%")
        if self.disk_percent > disk_threshold:
            overloaded.append(f"Disk: {self.disk_percent:.1f}%")

        return len(overloaded) > 0, overloaded

    def __str__(self) -> str:
        return (
            f"CPU: {self.cpu_percent:.1f}%, "
            f"GPU: {self.gpu_percent:.1f}%, "
            f"GPU Mem: {self.gpu_memory_percent:.1f}%, "
            f"RAM: {self.ram_percent:.1f}%, "
            f"Disk: {self.disk_percent:.1f}%"
        )


def get_cpu_percent() -> float:
    """Get current CPU utilization percentage."""
    try:
        # Use /proc/stat on Linux
        if os.path.exists("/proc/stat"):
            with open("/proc/stat") as f:
                line = f.readline()
            fields = line.split()[1:8]
            idle = int(fields[3])
            total = sum(int(f) for f in fields)
            # This is instantaneous, for better accuracy would need to sample over time
            # For now, use load average as proxy
            load = os.getloadavg()[0]
            cpu_count = os.cpu_count() or 1
            return min(100.0, (load / cpu_count) * 100)
        else:
            # macOS/other - use load average
            load = os.getloadavg()[0]
            cpu_count = os.cpu_count() or 1
            return min(100.0, (load / cpu_count) * 100)
    except Exception:
        return 0.0


def get_gpu_status() -> Tuple[float, float]:
    """Get GPU utilization and memory percentage.

    Returns:
        Tuple of (gpu_percent, gpu_memory_percent)
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            max_util = 0.0
            max_mem = 0.0
            for line in lines:
                parts = line.split(",")
                if len(parts) >= 3:
                    util = float(parts[0].strip())
                    mem_used = float(parts[1].strip())
                    mem_total = float(parts[2].strip())
                    mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0
                    max_util = max(max_util, util)
                    max_mem = max(max_mem, mem_pct)
            return max_util, max_mem
    except Exception:
        pass
    return 0.0, 0.0


def get_ram_percent() -> float:
    """Get RAM utilization percentage."""
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            mem_info = {}
            for line in lines:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = int(parts[1].strip().split()[0])
                    mem_info[key] = value

            total = mem_info.get("MemTotal", 1)
            available = mem_info.get("MemAvailable", mem_info.get("MemFree", 0))
            used = total - available
            return (used / total) * 100 if total > 0 else 0.0
        else:
            # macOS fallback
            import subprocess
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Simplified - just return 50% as fallback
            return 50.0
    except Exception:
        return 0.0


def get_disk_percent(path: str = "/") -> float:
    """Get disk utilization percentage for the given path."""
    try:
        usage = shutil.disk_usage(path)
        return (usage.used / usage.total) * 100 if usage.total > 0 else 0.0
    except Exception:
        return 0.0


def get_resource_status(disk_path: str = "/") -> ResourceStatus:
    """Get current resource status."""
    gpu_util, gpu_mem = get_gpu_status()
    return ResourceStatus(
        cpu_percent=get_cpu_percent(),
        gpu_percent=gpu_util,
        gpu_memory_percent=gpu_mem,
        ram_percent=get_ram_percent(),
        disk_percent=get_disk_percent(disk_path),
    )


def check_resources(
    cpu_threshold: float = DEFAULT_CPU_THRESHOLD,
    gpu_threshold: float = DEFAULT_GPU_THRESHOLD,
    ram_threshold: float = DEFAULT_RAM_THRESHOLD,
    disk_threshold: float = DEFAULT_DISK_THRESHOLD,
    disk_path: str = "/",
) -> bool:
    """Quick check if resources are available.

    Returns:
        True if resources are below thresholds, False if overloaded
    """
    status = get_resource_status(disk_path)
    is_overloaded, _ = status.is_overloaded(
        cpu_threshold, gpu_threshold, ram_threshold, disk_threshold
    )
    return not is_overloaded


class ResourceLimiter:
    """Resource limiter with automatic backoff.

    Usage:
        limiter = ResourceLimiter()

        # Check before work
        await limiter.wait_for_resources()

        # Or use context manager
        async with limiter.acquire("my_task"):
            do_work()
    """

    def __init__(
        self,
        cpu_threshold: float = DEFAULT_CPU_THRESHOLD,
        gpu_threshold: float = DEFAULT_GPU_THRESHOLD,
        ram_threshold: float = DEFAULT_RAM_THRESHOLD,
        disk_threshold: float = DEFAULT_DISK_THRESHOLD,
        disk_path: str = "/",
        max_concurrent_tasks: int = 0,  # 0 = unlimited
    ):
        self.cpu_threshold = cpu_threshold
        self.gpu_threshold = gpu_threshold
        self.ram_threshold = ram_threshold
        self.disk_threshold = disk_threshold
        self.disk_path = disk_path
        self.max_concurrent_tasks = max_concurrent_tasks

        self._active_tasks: Dict[str, float] = {}
        self._backoff_seconds = MIN_BACKOFF_SECONDS
        self._lock = asyncio.Lock()

    def get_status(self) -> ResourceStatus:
        """Get current resource status."""
        return get_resource_status(self.disk_path)

    def is_overloaded(self) -> Tuple[bool, List[str]]:
        """Check if resources are overloaded."""
        status = self.get_status()
        return status.is_overloaded(
            self.cpu_threshold,
            self.gpu_threshold,
            self.ram_threshold,
            self.disk_threshold,
        )

    async def wait_for_resources(self, task_name: str = "unknown") -> None:
        """Wait until resources are available."""
        while True:
            is_overloaded, reasons = self.is_overloaded()

            # Also check max concurrent tasks
            if self.max_concurrent_tasks > 0:
                if len(self._active_tasks) >= self.max_concurrent_tasks:
                    is_overloaded = True
                    reasons.append(f"Max tasks: {len(self._active_tasks)}/{self.max_concurrent_tasks}")

            if not is_overloaded:
                self._backoff_seconds = MIN_BACKOFF_SECONDS
                return

            logger.warning(
                f"[ResourceLimiter] Task '{task_name}' backing off for {self._backoff_seconds}s. "
                f"Overloaded: {', '.join(reasons)}"
            )

            await asyncio.sleep(self._backoff_seconds)

            # Exponential backoff
            self._backoff_seconds = min(
                self._backoff_seconds * BACKOFF_MULTIPLIER,
                MAX_BACKOFF_SECONDS
            )

    async def acquire(self, task_name: str):
        """Context manager to acquire resources for a task."""
        return _ResourceContext(self, task_name)

    def _register_task(self, task_name: str) -> None:
        """Register an active task."""
        self._active_tasks[task_name] = time.time()

    def _unregister_task(self, task_name: str) -> None:
        """Unregister a completed task."""
        self._active_tasks.pop(task_name, None)


class _ResourceContext:
    """Context manager for resource acquisition."""

    def __init__(self, limiter: ResourceLimiter, task_name: str):
        self.limiter = limiter
        self.task_name = task_name

    async def __aenter__(self):
        await self.limiter.wait_for_resources(self.task_name)
        self.limiter._register_task(self.task_name)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.limiter._unregister_task(self.task_name)
        return False


# Synchronous versions for non-async code
def wait_for_resources_sync(
    task_name: str = "unknown",
    cpu_threshold: float = DEFAULT_CPU_THRESHOLD,
    gpu_threshold: float = DEFAULT_GPU_THRESHOLD,
    ram_threshold: float = DEFAULT_RAM_THRESHOLD,
    disk_threshold: float = DEFAULT_DISK_THRESHOLD,
    disk_path: str = "/",
    max_wait_seconds: float = 600,
) -> bool:
    """Synchronous version of wait_for_resources.

    Returns:
        True if resources became available, False if max wait exceeded
    """
    backoff = MIN_BACKOFF_SECONDS
    total_waited = 0.0

    while total_waited < max_wait_seconds:
        status = get_resource_status(disk_path)
        is_overloaded, reasons = status.is_overloaded(
            cpu_threshold, gpu_threshold, ram_threshold, disk_threshold
        )

        if not is_overloaded:
            return True

        logger.warning(
            f"[ResourceLimiter] Task '{task_name}' backing off for {backoff}s. "
            f"Overloaded: {', '.join(reasons)}"
        )

        time.sleep(backoff)
        total_waited += backoff
        backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_SECONDS)

    logger.error(f"[ResourceLimiter] Task '{task_name}' exceeded max wait time")
    return False


# Decorator for functions that should respect resource limits
def respect_resource_limits(
    task_name: Optional[str] = None,
    cpu_threshold: float = DEFAULT_CPU_THRESHOLD,
    gpu_threshold: float = DEFAULT_GPU_THRESHOLD,
    ram_threshold: float = DEFAULT_RAM_THRESHOLD,
    disk_threshold: float = DEFAULT_DISK_THRESHOLD,
):
    """Decorator that waits for resources before executing function."""
    def decorator(func: Callable):
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = task_name or func.__name__
            if wait_for_resources_sync(
                name, cpu_threshold, gpu_threshold, ram_threshold, disk_threshold
            ):
                return func(*args, **kwargs)
            else:
                raise RuntimeError(f"Resources not available for {name}")

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = task_name or func.__name__
            limiter = ResourceLimiter(
                cpu_threshold, gpu_threshold, ram_threshold, disk_threshold
            )
            await limiter.wait_for_resources(name)
            return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check resource utilization")
    parser.add_argument("--watch", action="store_true", help="Continuously monitor")
    parser.add_argument("--interval", type=float, default=5, help="Watch interval")
    args = parser.parse_args()

    if args.watch:
        print("Monitoring resources (Ctrl+C to stop)...")
        try:
            while True:
                status = get_resource_status()
                is_overloaded, reasons = status.is_overloaded()
                overload_str = f" OVERLOADED: {', '.join(reasons)}" if is_overloaded else ""
                print(f"{status}{overload_str}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped")
    else:
        status = get_resource_status()
        print(f"Resource Status: {status}")
        is_overloaded, reasons = status.is_overloaded()
        if is_overloaded:
            print(f"OVERLOADED: {', '.join(reasons)}")
        else:
            print("All resources within limits")
