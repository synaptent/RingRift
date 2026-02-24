"""Unified Resource Guard - Pre-operation resource checks.

THIS IS THE CANONICAL SOURCE for resource checking across the codebase.
Do NOT use shutil.disk_usage(), psutil.virtual_memory(), or psutil.cpu_percent()
directly in scripts. Use this module instead for consistent limits and metrics.

Provides simple functions for scripts to check resource availability
before performing operations. Enforces the shared resource limits below
(CPU/GPU 80%, memory 90%, disk 95%).

Usage:
    from app.utils.resource_guard import (
        check_disk_space,
        check_memory,
        check_cpu,
        check_gpu_memory,
        can_proceed,
        wait_for_resources,
        LIMITS,  # Access limit constants
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

    # Access limits
    from app.utils.resource_guard import LIMITS
    print(f"Max disk usage: {LIMITS.DISK_MAX_PERCENT}%")

Why use this module:
    - Consistent resource utilization limits across all scripts
    - Prometheus metrics integration for monitoring
    - Graceful degradation support
    - Single source of truth for resource thresholds
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.utils.optional_imports import (
    PROMETHEUS_AVAILABLE as HAS_PROMETHEUS,
    Counter,
    Gauge,
)
from app.utils.time_constants import SECONDS_PER_DAY

logger = logging.getLogger(__name__)

__all__ = [
    "DISK_PRESSURE_THRESHOLDS",
    "LIMITS",
    "PRESSURE_THRESHOLDS",
    # Disk pressure management
    "DiskPressureLevel",
    # Disk monitoring
    "DiskPressureMonitor",
    # Memory pressure management
    "MemoryPressureLevel",
    # Memory monitoring
    "MemoryPressureMonitor",
    # Limits and configuration
    "ResourceLimits",
    "adjust_oom_score",
    "can_proceed",
    "coordinator_resource_gate",
    "check_cpu",
    "check_disk_and_cleanup",
    "check_disk_for_write",
    # Core check functions
    "check_disk_space",
    "check_gpu_memory",
    "check_memory",
    "check_memory_and_cleanup",
    "cleanup_old_checkpoints",
    "cleanup_old_games",
    # Cleanup functions
    "cleanup_old_logs",
    "cleanup_temp_files",
    # GPU utilities
    "clear_gpu_memory",
    # Utility
    "get_ai_service_root",
    "get_cpu_usage",
    "get_disk_pressure_level",
    # Usage info functions
    "get_disk_usage",
    "get_gpu_memory_usage",
    "get_memory_pressure_level",
    "get_memory_usage",
    "get_oom_score",
    "get_psi_memory_pressure",
    "get_resource_status",
    "register_oom_signal_handler",
    "start_disk_monitor",
    "start_memory_monitor",
    "stop_disk_monitor",
    "stop_memory_monitor",
    "trigger_disk_cleanup",
    "trigger_memory_cleanup",
    "wait_for_resources",
]

# Initialize Prometheus metrics if available
if HAS_PROMETHEUS:
    PROM_CPU_USAGE = Gauge(
        'ringrift_resource_cpu_percent',
        'Current CPU usage percent'
    )
    PROM_MEMORY_USAGE = Gauge(
        'ringrift_resource_memory_percent',
        'Current memory usage percent'
    )
    PROM_DISK_USAGE = Gauge(
        'ringrift_resource_disk_percent',
        'Current disk usage percent'
    )
    PROM_GPU_USAGE = Gauge(
        'ringrift_resource_gpu_percent',
        'Current GPU memory usage percent'
    )
    PROM_RESOURCE_LIMIT = Gauge(
        'ringrift_resource_limit_percent',
        'Resource limit threshold',
        ['resource_type']
    )
    PROM_RESOURCE_OK = Gauge(
        'ringrift_resource_available',
        'Whether resources are available (1=ok, 0=exceeded)',
        ['resource_type']
    )
    PROM_RESOURCE_WAIT_COUNT = Counter(
        'ringrift_resource_wait_total',
        'Number of times we had to wait for resources'
    )
    PROM_RESOURCE_BACKOFF_COUNT = Counter(
        'ringrift_resource_backoff_total',
        'Number of resource backoff events',
        ['resource_type']
    )
    PROM_DEGRADATION_LEVEL = Gauge(
        'ringrift_resource_degradation_level',
        'Current graceful degradation level (0=normal, 1=light, 2=moderate, 3=heavy, 4=critical)'
    )
    # Additional metrics with _used_ naming for alert rules
    PROM_DISK_USED = Gauge(
        'ringrift_resource_disk_used_percent',
        'Current disk usage percent (for alerting)'
    )
    PROM_MEMORY_USED = Gauge(
        'ringrift_resource_memory_used_percent',
        'Current memory usage percent (for alerting)'
    )
    PROM_CPU_USED = Gauge(
        'ringrift_resource_cpu_used_percent',
        'Current CPU usage percent (for alerting)'
    )
    PROM_GPU_USED = Gauge(
        'ringrift_resource_gpu_used_percent',
        'Current GPU usage percent (for alerting)'
    )

    # Pressure level metrics (2025-12)
    PROM_MEMORY_PRESSURE_LEVEL = Gauge(
        'ringrift_memory_pressure_level',
        'Current memory pressure level (0=normal, 1=warning, 2=elevated, 3=critical, 4=emergency)'
    )
    PROM_DISK_PRESSURE_LEVEL = Gauge(
        'ringrift_disk_pressure_level',
        'Current disk pressure level (0=normal, 1=warning, 2=elevated, 3=critical, 4=emergency)'
    )
    PROM_PRESSURE_CLEANUP_TOTAL = Counter(
        'ringrift_pressure_cleanup_total',
        'Total cleanup operations triggered by pressure',
        ['resource_type', 'cleanup_type']
    )
    PROM_PRESSURE_CRITICAL_EVENTS = Counter(
        'ringrift_pressure_critical_events_total',
        'Critical pressure events requiring intervention',
        ['resource_type']
    )
    PROM_OOM_SCORE_ADJUSTED = Gauge(
        'ringrift_oom_score_adjusted',
        'Current OOM score adjustment value'
    )

# ============================================
# Resource Limits - shared utilization thresholds
# ============================================

@dataclass(frozen=True)
class ResourceLimits:
    """Unified resource limits (CPU/GPU 80%, memory 90%, disk 95%).

    Note: For cluster-wide disk thresholds, see app.config.thresholds:
    DISK_SYNC_TARGET_PERCENT (70), DISK_PRODUCTION_HALT_PERCENT (85),
    DISK_CRITICAL_PERCENT (90). These per-operation limits are intentionally
    higher to avoid aborting in-progress work.
    """
    # Disk at 95% - increased to allow selfplay generation with low headroom
    DISK_MAX_PERCENT: float = 95.0
    DISK_WARN_PERCENT: float = 90.0

    # CPU/GPU/Memory at 80% hard limit
    CPU_MAX_PERCENT: float = 80.0
    CPU_WARN_PERCENT: float = 70.0

    GPU_MAX_PERCENT: float = 80.0
    GPU_WARN_PERCENT: float = 70.0

    MEMORY_MAX_PERCENT: float = 90.0
    MEMORY_WARN_PERCENT: float = 80.0

    # Load average limit
    LOAD_MAX_FACTOR: float = 1.5  # load > cpus * this = overloaded


LIMITS = ResourceLimits()


# ============================================
# Disk Space Checks
# ============================================

def get_disk_usage(path: str | None = None) -> tuple[float, float, float]:
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
    except (OSError, PermissionError) as e:
        # Disk access failures (path not found, permission denied, I/O errors)
        logger.warning(f"Could not check disk usage: {e}")
        return 0.0, 999.0, 999.0  # Assume OK on error


def check_disk_space(
    required_gb: float = 2.0,
    path: str | None = None,
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
    path: str | None = None,
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

def get_memory_usage() -> tuple[float, float, float]:
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
    except (OSError, AttributeError) as e:
        # OS-level memory access failure or psutil attribute missing
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

def get_cpu_usage() -> tuple[float, float, int]:
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
    except (OSError, ValueError) as e:
        # OS-level CPU info failure or invalid numeric conversion
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

def get_gpu_memory_usage(device_id: int = 0) -> tuple[float, float, float]:
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
    except (RuntimeError, ValueError) as e:
        # CUDA runtime errors (OOM, invalid device) or value conversion errors
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


def clear_gpu_memory() -> None:
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
# Memory Pressure Detection & OOM Prevention
# ============================================

@dataclass(frozen=True)
class MemoryPressureLevel:
    """Memory pressure levels for graduated response."""
    NORMAL = 0      # < 70% - normal operation
    WARNING = 1     # 70-80% - start cleanup, reduce new allocations
    ELEVATED = 2    # 80-85% - aggressive cleanup, pause non-critical work
    CRITICAL = 3    # 85-90% - emergency cleanup, stop new operations
    EMERGENCY = 4   # > 90% - trigger OOM prevention measures


PRESSURE_THRESHOLDS = (70.0, 80.0, 85.0, 90.0)


def get_memory_pressure_level() -> int:
    """Get current memory pressure level.

    Returns:
        MemoryPressureLevel value (0-4)
    """
    used_percent, _, _ = get_memory_usage()

    if used_percent >= 90.0:
        level = MemoryPressureLevel.EMERGENCY
    elif used_percent >= 85.0:
        level = MemoryPressureLevel.CRITICAL
    elif used_percent >= 80.0:
        level = MemoryPressureLevel.ELEVATED
    elif used_percent >= 70.0:
        level = MemoryPressureLevel.WARNING
    else:
        level = MemoryPressureLevel.NORMAL

    # Update Prometheus metric
    if HAS_PROMETHEUS:
        PROM_MEMORY_PRESSURE_LEVEL.set(level)

    return level


def get_psi_memory_pressure() -> dict | None:
    """Get Linux PSI (Pressure Stall Information) metrics.

    PSI provides system-wide resource pressure information.
    Only available on Linux 4.20+ with cgroup v2.

    Returns:
        Dict with 'some' and 'full' pressure percentages, or None if unavailable.
    """
    psi_file = Path("/proc/pressure/memory")
    if not psi_file.exists():
        return None

    try:
        content = psi_file.read_text()
        result = {}
        for line in content.strip().split('\n'):
            parts = line.split()
            if parts[0] == 'some':
                # Parse: some avg10=0.00 avg60=0.00 avg300=0.00 total=0
                for part in parts[1:]:
                    if part.startswith('avg10='):
                        result['some_avg10'] = float(part.split('=')[1])
                    elif part.startswith('avg60='):
                        result['some_avg60'] = float(part.split('=')[1])
            elif parts[0] == 'full':
                for part in parts[1:]:
                    if part.startswith('avg10='):
                        result['full_avg10'] = float(part.split('=')[1])
                    elif part.startswith('avg60='):
                        result['full_avg60'] = float(part.split('=')[1])
        return result if result else None
    except (OSError, ValueError, IndexError) as e:
        # File read failure, invalid float conversion, or malformed PSI data
        logger.debug(f"Could not read PSI metrics: {e}")
        return None


def adjust_oom_score(score_adj: int = 500) -> bool:
    """Adjust OOM killer priority for current process.

    Higher score_adj = more likely to be killed by OOM killer.
    Range is -1000 (never kill) to +1000 (kill first).

    For GPU training, we want to be killed before system daemons
    but after less important processes. 500 is a reasonable value.

    Args:
        score_adj: OOM score adjustment (-1000 to 1000)

    Returns:
        True if adjustment was successful.
    """
    oom_adj_path = Path(f"/proc/{os.getpid()}/oom_score_adj")
    if not oom_adj_path.exists():
        return False

    try:
        oom_adj_path.write_text(str(score_adj))
        logger.info(f"Set OOM score adjustment to {score_adj}")
        return True
    except PermissionError:
        logger.debug("Cannot adjust OOM score (requires root)")
        return False
    except (OSError, ValueError) as e:
        # File write failure or invalid score_adj value
        logger.debug(f"Could not adjust OOM score: {e}")
        return False


def get_oom_score() -> int | None:
    """Get current OOM score for this process.

    Returns:
        Current OOM score (0-1000), or None if unavailable.
    """
    try:
        oom_score_path = Path(f"/proc/{os.getpid()}/oom_score")
        if oom_score_path.exists():
            return int(oom_score_path.read_text().strip())
        return None
    except (OSError, ValueError):
        # File read failure or invalid integer conversion
        return None


def trigger_memory_cleanup(level: int) -> int:
    """Trigger memory cleanup based on pressure level.

    Args:
        level: MemoryPressureLevel value

    Returns:
        Estimated MB freed.
    """
    import gc
    freed_mb = 0

    # Level 1+: Python garbage collection
    if level >= MemoryPressureLevel.WARNING:
        collected = gc.collect()
        freed_mb += collected * 0.001  # Rough estimate

    # Level 2+: Clear GPU cache
    if level >= MemoryPressureLevel.ELEVATED:
        clear_gpu_memory()
        freed_mb += 100  # GPU cache can be significant

    # Level 3+: Aggressive cleanup
    if level >= MemoryPressureLevel.CRITICAL:
        # Run multiple GC generations
        for _ in range(3):
            gc.collect()

        # Clear various caches
        try:
            import functools
            if hasattr(functools, '_lru_cache_wrapper'):
                # Can't easily clear all LRU caches, but we try
                pass
        except (ImportError, AttributeError):
            # functools not available or missing attributes
            pass

        freed_mb += 50

    # Level 4: Emergency measures
    if level >= MemoryPressureLevel.EMERGENCY:
        logger.warning("EMERGENCY memory pressure - aggressive cleanup triggered")

        # Drop Python caches
        try:
            import importlib
            importlib.invalidate_caches()
        except (ImportError, AttributeError):
            # importlib not available or missing invalidate_caches
            pass

        # Clear more caches
        try:
            import sys
            sys.intern('')  # Hint to clear interned strings
        except (AttributeError, TypeError):
            # sys.intern not available or invalid argument
            pass

        # Final GC sweep
        gc.collect()
        freed_mb += 100

    # Emit Prometheus metrics
    if HAS_PROMETHEUS:
        PROM_PRESSURE_CLEANUP_TOTAL.labels(resource_type="memory", cleanup_type="gc").inc()
        if level >= MemoryPressureLevel.CRITICAL:
            PROM_PRESSURE_CRITICAL_EVENTS.labels(resource_type="memory").inc()

    logger.info(f"Memory cleanup at level {level}: ~{freed_mb:.0f}MB freed (estimated)")
    return int(freed_mb)


def check_memory_and_cleanup(
    required_gb: float = 1.0,
    auto_cleanup: bool = True,
) -> tuple[bool, int]:
    """Check memory and optionally trigger cleanup if needed.

    Args:
        required_gb: Minimum memory required in GB
        auto_cleanup: Whether to trigger cleanup on pressure

    Returns:
        Tuple of (memory_ok, pressure_level)
    """
    level = get_memory_pressure_level()

    if auto_cleanup and level >= MemoryPressureLevel.WARNING:
        trigger_memory_cleanup(level)
        # Re-check after cleanup
        level = get_memory_pressure_level()

    memory_ok = check_memory(required_gb, log_warning=False)
    return memory_ok, level


def register_oom_signal_handler() -> bool:
    """Register signal handlers for graceful OOM handling.

    On Linux, SIGTERM is often sent before SIGKILL during OOM.
    This gives us a brief window to save state.

    Returns:
        True if handlers were registered.
    """
    import signal

    def oom_handler(signum, frame):
        logger.critical("OOM signal received - attempting graceful shutdown")
        # Trigger emergency cleanup
        trigger_memory_cleanup(MemoryPressureLevel.EMERGENCY)
        # Let the default handler run
        signal.default_int_handler(signum, frame)

    try:
        signal.signal(signal.SIGTERM, oom_handler)
        return True
    except (OSError, ValueError, RuntimeError) as e:
        # Signal registration failure (invalid signal, platform unsupported, runtime error)
        logger.debug(f"Could not register OOM signal handler: {e}")
        return False


class MemoryPressureMonitor:
    """Background monitor for memory pressure with automatic cleanup.

    Usage:
        monitor = MemoryPressureMonitor()
        monitor.start()

        # ... do work ...

        monitor.stop()
    """

    def __init__(
        self,
        check_interval: float = 5.0,
        auto_cleanup: bool = True,
        on_critical: Callable | None = None,
    ):
        """
        Args:
            check_interval: Seconds between pressure checks
            auto_cleanup: Trigger cleanup on elevated pressure
            on_critical: Callback when reaching critical level
        """
        self.check_interval = check_interval
        self.auto_cleanup = auto_cleanup
        self.on_critical = on_critical
        self._running = False
        self._thread = None
        self._last_level = MemoryPressureLevel.NORMAL

    def start(self):
        """Start the background monitor."""
        if self._running:
            return

        import threading
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Memory pressure monitor started")

    def stop(self):
        """Stop the background monitor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                level = get_memory_pressure_level()

                # Log transitions
                if level != self._last_level:
                    level_names = ['NORMAL', 'WARNING', 'ELEVATED', 'CRITICAL', 'EMERGENCY']
                    logger.info(
                        f"Memory pressure: {level_names[self._last_level]} -> {level_names[level]}"
                    )
                    self._last_level = level

                # Auto cleanup
                if self.auto_cleanup and level >= MemoryPressureLevel.ELEVATED:
                    trigger_memory_cleanup(level)

                # Critical callback
                if level >= MemoryPressureLevel.CRITICAL and self.on_critical:
                    try:
                        self.on_critical(level)
                    except (TypeError, RuntimeError, OSError) as e:
                        # Callback invocation error (wrong signature, runtime error, resource failure)
                        logger.warning(f"Critical callback error: {e}")

            except (OSError, MemoryError, RuntimeError) as e:
                # Memory check failure, out of memory, or runtime error
                logger.debug(f"Memory monitor error: {e}")

            time.sleep(self.check_interval)


# Global monitor instance (optional)
_memory_monitor: MemoryPressureMonitor | None = None


def start_memory_monitor(
    check_interval: float = 5.0,
    auto_cleanup: bool = True,
    on_critical: Callable | None = None,
) -> MemoryPressureMonitor:
    """Start the global memory pressure monitor.

    Returns:
        The monitor instance.
    """
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = MemoryPressureMonitor(
            check_interval=check_interval,
            auto_cleanup=auto_cleanup,
            on_critical=on_critical,
        )
        _memory_monitor.start()
    return _memory_monitor


def stop_memory_monitor():
    """Stop the global memory pressure monitor."""
    global _memory_monitor
    if _memory_monitor is not None:
        _memory_monitor.stop()
        _memory_monitor = None


# ============================================
# Disk Space Cleanup Automation
# ============================================

@dataclass(frozen=True)
class DiskPressureLevel:
    """Disk pressure levels for graduated cleanup."""
    NORMAL = 0      # < 70% - normal operation
    WARNING = 1     # 70-75% - start cleanup, archive old files
    ELEVATED = 2    # 75-80% - aggressive cleanup, remove old logs
    CRITICAL = 3    # 80-85% - emergency cleanup, remove old models
    EMERGENCY = 4   # > 85% - remove all non-essential files


# Pressure gradient thresholds for DiskPressureLevel enum.
# See app.config.thresholds for canonical hard cutoffs (70/85/90%).
DISK_PRESSURE_THRESHOLDS = (70.0, 75.0, 80.0, 85.0)


def get_disk_pressure_level(path: str | None = None) -> int:
    """Get current disk pressure level.

    Returns:
        DiskPressureLevel value (0-4)
    """
    used_percent, _, _ = get_disk_usage(path)

    if used_percent >= 85.0:
        level = DiskPressureLevel.EMERGENCY
    elif used_percent >= 80.0:
        level = DiskPressureLevel.CRITICAL
    elif used_percent >= 75.0:
        level = DiskPressureLevel.ELEVATED
    elif used_percent >= 70.0:
        level = DiskPressureLevel.WARNING
    else:
        level = DiskPressureLevel.NORMAL

    # Update Prometheus metric
    if HAS_PROMETHEUS:
        PROM_DISK_PRESSURE_LEVEL.set(level)

    return level


def get_ai_service_root() -> Path:
    """Get the AI service root directory."""
    return Path(__file__).parent.parent.parent


def cleanup_old_logs(max_age_days: int = 7) -> int:
    """Clean up old log files.

    Args:
        max_age_days: Delete logs older than this many days

    Returns:
        Number of files deleted.
    """
    import time
    logs_dir = get_ai_service_root() / "logs"
    if not logs_dir.exists():
        return 0

    cutoff_time = time.time() - (max_age_days * SECONDS_PER_DAY)
    deleted = 0

    for log_file in logs_dir.glob("**/*.log*"):
        try:
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                deleted += 1
        except (OSError, PermissionError):
            # File access/deletion failure (missing file, permission denied)
            pass

    if deleted > 0:
        logger.info(f"Deleted {deleted} old log files")
    return deleted


def cleanup_old_checkpoints(keep_per_config: int = 5, dry_run: bool = False) -> tuple[int, int]:
    """Clean up old model checkpoints, keeping the most recent per config.

    Args:
        keep_per_config: Number of models to keep per board config
        dry_run: If True, only report what would be deleted

    Returns:
        Tuple of (files_deleted, bytes_freed)
    """
    import re
    from collections import defaultdict
    from datetime import datetime

    models_dir = get_ai_service_root() / "models"
    if not models_dir.exists():
        return 0, 0

    # Group models by config
    models_by_config = defaultdict(list)

    for model_file in models_dir.glob("*.pth"):
        filename = model_file.name

        # Skip best models
        if "best" in filename.lower():
            continue

        # Try to extract timestamp
        match = re.search(r'(\d{8}_\d{6})', filename)
        if not match:
            continue

        try:
            timestamp = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        except (ValueError, AttributeError):
            # Invalid timestamp format or missing match group
            continue

        # Extract board config
        config_match = re.search(r'(sq(?:uare)?(?:8|19))_(\d)p', filename, re.IGNORECASE)
        if config_match:
            config = f"{config_match.group(1).lower()}_{config_match.group(2)}p"
        elif "hex" in filename.lower():
            players_match = re.search(r'(\d)p', filename)
            config = f"hex_{players_match.group(1) if players_match else '2'}p"
        else:
            config = "unknown"

        models_by_config[config].append((model_file, timestamp))

    # For each config, keep only the most recent
    files_deleted = 0
    bytes_freed = 0

    for _config, models in models_by_config.items():
        # Sort by timestamp, newest first
        models.sort(key=lambda x: x[1], reverse=True)

        # Delete old models
        for model_file, _ in models[keep_per_config:]:
            try:
                size = model_file.stat().st_size
                if not dry_run:
                    model_file.unlink()
                files_deleted += 1
                bytes_freed += size
            except (OSError, PermissionError):
                # File access/deletion failure
                pass

    if files_deleted > 0:
        action = "Would delete" if dry_run else "Deleted"
        logger.info(f"{action} {files_deleted} old checkpoints ({bytes_freed / 1024 / 1024:.1f}MB)")

    return files_deleted, bytes_freed


def cleanup_temp_files() -> int:
    """Clean up temporary files (.tmp, .part, etc).

    Returns:
        Number of files deleted.
    """
    root = get_ai_service_root()
    deleted = 0

    patterns = ["**/*.tmp", "**/*.part", "**/*.temp", "**/~*", "**/*.pyc"]

    for pattern in patterns:
        for temp_file in root.glob(pattern):
            try:
                # Skip if modified in the last hour (might be in use)
                if (time.time() - temp_file.stat().st_mtime) > 3600:
                    temp_file.unlink()
                    deleted += 1
            except (OSError, PermissionError):
                # File access/deletion failure
                pass

    if deleted > 0:
        logger.info(f"Deleted {deleted} temporary files")
    return deleted


def cleanup_old_games(keep_days: int = 30) -> tuple[int, int]:
    """Clean up old game databases beyond retention period.

    Args:
        keep_days: Keep games from the last N days

    Returns:
        Tuple of (files_deleted, bytes_freed)
    """
    import time
    games_dir = get_ai_service_root() / "data" / "games"
    if not games_dir.exists():
        return 0, 0

    cutoff_time = time.time() - (keep_days * SECONDS_PER_DAY)
    files_deleted = 0
    bytes_freed = 0

    # Only clean up synced subdirectories, not main databases
    synced_dir = games_dir / "synced"
    if synced_dir.exists():
        for db_file in synced_dir.glob("**/*.db"):
            try:
                if db_file.stat().st_mtime < cutoff_time:
                    size = db_file.stat().st_size
                    db_file.unlink()
                    files_deleted += 1
                    bytes_freed += size
            except (OSError, PermissionError):
                # File access/deletion failure
                pass

    if files_deleted > 0:
        logger.info(f"Deleted {files_deleted} old game files ({bytes_freed / 1024 / 1024:.1f}MB)")

    return files_deleted, bytes_freed


def trigger_disk_cleanup(level: int, dry_run: bool = False) -> int:
    """Trigger disk cleanup based on pressure level.

    Args:
        level: DiskPressureLevel value
        dry_run: If True, only report what would be done

    Returns:
        Estimated MB freed.
    """
    freed_mb = 0

    # Level 1+: Clean temporary files
    if level >= DiskPressureLevel.WARNING:
        deleted = cleanup_temp_files()
        freed_mb += deleted * 0.1  # Rough estimate

    # Level 2+: Clean old logs
    if level >= DiskPressureLevel.ELEVATED:
        deleted = cleanup_old_logs(max_age_days=3)  # More aggressive at level 2
        freed_mb += deleted * 1  # ~1MB per log file

    # Level 3+: Clean old checkpoints
    if level >= DiskPressureLevel.CRITICAL:
        _files, bytes_freed = cleanup_old_checkpoints(keep_per_config=3, dry_run=dry_run)
        freed_mb += bytes_freed / (1024 * 1024)

    # Level 4: Emergency cleanup
    if level >= DiskPressureLevel.EMERGENCY:
        logger.warning("EMERGENCY disk pressure - aggressive cleanup triggered")

        # More aggressive log cleanup
        cleanup_old_logs(max_age_days=1)

        # More aggressive checkpoint cleanup
        _files, bytes_freed = cleanup_old_checkpoints(keep_per_config=2, dry_run=dry_run)
        freed_mb += bytes_freed / (1024 * 1024)

        # Clean old games
        _files, bytes_freed = cleanup_old_games(keep_days=14)
        freed_mb += bytes_freed / (1024 * 1024)

    # Emit Prometheus metrics
    if HAS_PROMETHEUS:
        PROM_PRESSURE_CLEANUP_TOTAL.labels(resource_type="disk", cleanup_type="files").inc()
        if level >= DiskPressureLevel.CRITICAL:
            PROM_PRESSURE_CRITICAL_EVENTS.labels(resource_type="disk").inc()

    logger.info(f"Disk cleanup at level {level}: ~{freed_mb:.0f}MB freed")
    return int(freed_mb)


def check_disk_and_cleanup(
    required_gb: float = 2.0,
    auto_cleanup: bool = True,
    path: str | None = None,
) -> tuple[bool, int]:
    """Check disk space and optionally trigger cleanup if needed.

    Args:
        required_gb: Minimum disk space required in GB
        auto_cleanup: Whether to trigger cleanup on pressure
        path: Path to check

    Returns:
        Tuple of (disk_ok, pressure_level)
    """
    level = get_disk_pressure_level(path)

    if auto_cleanup and level >= DiskPressureLevel.WARNING:
        trigger_disk_cleanup(level)
        # Re-check after cleanup
        level = get_disk_pressure_level(path)

    disk_ok = check_disk_space(required_gb, path, log_warning=False)
    return disk_ok, level


class DiskPressureMonitor:
    """Background monitor for disk pressure with automatic cleanup.

    Usage:
        monitor = DiskPressureMonitor()
        monitor.start()

        # ... do work ...

        monitor.stop()
    """

    def __init__(
        self,
        check_interval: float = 60.0,  # Check every minute
        auto_cleanup: bool = True,
        on_critical: Callable | None = None,
        path: str | None = None,
    ):
        """
        Args:
            check_interval: Seconds between pressure checks
            auto_cleanup: Trigger cleanup on elevated pressure
            on_critical: Callback when reaching critical level
            path: Path to monitor
        """
        self.check_interval = check_interval
        self.auto_cleanup = auto_cleanup
        self.on_critical = on_critical
        self.path = path
        self._running = False
        self._thread = None
        self._last_level = DiskPressureLevel.NORMAL

    def start(self):
        """Start the background monitor."""
        if self._running:
            return

        import threading
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Disk pressure monitor started")

    def stop(self):
        """Stop the background monitor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                level = get_disk_pressure_level(self.path)

                # Log transitions
                if level != self._last_level:
                    level_names = ['NORMAL', 'WARNING', 'ELEVATED', 'CRITICAL', 'EMERGENCY']
                    logger.info(
                        f"Disk pressure: {level_names[self._last_level]} -> {level_names[level]}"
                    )
                    self._last_level = level

                # Auto cleanup
                if self.auto_cleanup and level >= DiskPressureLevel.ELEVATED:
                    trigger_disk_cleanup(level)

                # Critical callback
                if level >= DiskPressureLevel.CRITICAL and self.on_critical:
                    try:
                        self.on_critical(level)
                    except (TypeError, RuntimeError, OSError) as e:
                        # Callback invocation error (wrong signature, runtime error, resource failure)
                        logger.warning(f"Critical callback error: {e}")

            except (OSError, RuntimeError) as e:
                # Disk check failure or runtime error
                logger.debug(f"Disk monitor error: {e}")

            time.sleep(self.check_interval)


# Global disk monitor instance
_disk_monitor: DiskPressureMonitor | None = None


def start_disk_monitor(
    check_interval: float = 60.0,
    auto_cleanup: bool = True,
    on_critical: Callable | None = None,
) -> DiskPressureMonitor:
    """Start the global disk pressure monitor.

    Returns:
        The monitor instance.
    """
    global _disk_monitor
    if _disk_monitor is None:
        _disk_monitor = DiskPressureMonitor(
            check_interval=check_interval,
            auto_cleanup=auto_cleanup,
            on_critical=on_critical,
        )
        _disk_monitor.start()
    return _disk_monitor


def stop_disk_monitor():
    """Stop the global disk pressure monitor."""
    global _disk_monitor
    if _disk_monitor is not None:
        _disk_monitor.stop()
        _disk_monitor = None


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

    return not (check_gpu and not check_gpu_memory(gpu_required_gb, log_warning=False))


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
    # Increment Prometheus wait counter
    if HAS_PROMETHEUS:
        PROM_RESOURCE_WAIT_COUNT.inc()

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
# Coordinator Resource Gate
# ============================================

# February 2026: Coordinator machines run 45+ daemons simultaneously and are
# especially vulnerable to OOM / swap exhaustion.  Heavy operations (DB merges,
# NPZ exports, rsync transfers, backups) must NOT start when physical memory or
# disk headroom is below a safe threshold.  The default is 30 % free for both
# (configurable via environment variables).

_COORD_MIN_FREE_RAM_PCT = float(
    os.environ.get("RINGRIFT_COORDINATOR_MIN_FREE_RAM_PERCENT", "30")
)
_COORD_MIN_FREE_DISK_PCT = float(
    os.environ.get("RINGRIFT_COORDINATOR_MIN_FREE_DISK_PERCENT", "10")
)


def coordinator_resource_gate(operation: str) -> bool:
    """Return True if a heavy operation may proceed on a coordinator node.

    Checks:
        1. Physical RAM free >= RINGRIFT_COORDINATOR_MIN_FREE_RAM_PERCENT  (default 30 %)
        2. Disk free >= RINGRIFT_COORDINATOR_MIN_FREE_DISK_PERCENT  (default 30 %)

    The check is ONLY enforced when RINGRIFT_IS_COORDINATOR=true.  On worker
    nodes the function always returns True so callers don't need separate
    branching logic.

    Args:
        operation: Human-readable label for logging (e.g. "NPZ_EXPORT",
                   "CLUSTER_CONSOLIDATION").

    Returns:
        True if the operation may proceed, False if it should be skipped.
    """
    # Only enforce on coordinator nodes
    if os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower() != "true":
        return True

    # --- RAM check ---
    mem_used_pct, mem_avail_gb, mem_total_gb = get_memory_usage()
    mem_free_pct = 100.0 - mem_used_pct
    if mem_free_pct < _COORD_MIN_FREE_RAM_PCT:
        logger.warning(
            f"[coordinator_resource_gate] BLOCKED {operation}: "
            f"RAM free {mem_free_pct:.1f}% < {_COORD_MIN_FREE_RAM_PCT:.0f}% "
            f"({mem_avail_gb:.1f} GB / {mem_total_gb:.1f} GB available)"
        )
        return False

    # --- Disk check ---
    disk_used_pct, disk_avail_gb, disk_total_gb = get_disk_usage()
    disk_free_pct = 100.0 - disk_used_pct
    if disk_free_pct < _COORD_MIN_FREE_DISK_PCT:
        logger.warning(
            f"[coordinator_resource_gate] BLOCKED {operation}: "
            f"disk free {disk_free_pct:.1f}% < {_COORD_MIN_FREE_DISK_PCT:.0f}% "
            f"({disk_avail_gb:.1f} GB / {disk_total_gb:.1f} GB available)"
        )
        return False

    logger.debug(
        f"[coordinator_resource_gate] ALLOWED {operation}: "
        f"RAM free {mem_free_pct:.1f}%, disk free {disk_free_pct:.1f}%"
    )
    return True


# ============================================
# Resource Status Report
# ============================================

def get_resource_status(export_prometheus: bool = True) -> dict:
    """Get current resource status as a dictionary.

    Args:
        export_prometheus: Whether to export metrics to Prometheus

    Returns:
        Dict with resource usage and status
    """
    disk_pct, disk_avail, disk_total = get_disk_usage()
    mem_pct, mem_avail, mem_total = get_memory_usage()
    cpu_pct, load_per_cpu, cpu_count = get_cpu_usage()
    gpu_pct, gpu_avail, gpu_total = get_gpu_memory_usage()

    disk_ok = disk_pct < LIMITS.DISK_MAX_PERCENT
    mem_ok = mem_pct < LIMITS.MEMORY_MAX_PERCENT
    cpu_ok = cpu_pct < LIMITS.CPU_MAX_PERCENT and load_per_cpu < LIMITS.LOAD_MAX_FACTOR
    gpu_ok = gpu_pct < LIMITS.GPU_MAX_PERCENT or gpu_total == 0

    # Export to Prometheus if available
    if export_prometheus and HAS_PROMETHEUS:
        PROM_CPU_USAGE.set(cpu_pct)
        PROM_MEMORY_USAGE.set(mem_pct)
        PROM_DISK_USAGE.set(disk_pct)
        PROM_GPU_USAGE.set(gpu_pct)
        # Set limit thresholds (only once, but safe to repeat)
        PROM_RESOURCE_LIMIT.labels(resource_type='cpu').set(LIMITS.CPU_MAX_PERCENT)
        PROM_RESOURCE_LIMIT.labels(resource_type='memory').set(LIMITS.MEMORY_MAX_PERCENT)
        PROM_RESOURCE_LIMIT.labels(resource_type='disk').set(LIMITS.DISK_MAX_PERCENT)
        PROM_RESOURCE_LIMIT.labels(resource_type='gpu').set(LIMITS.GPU_MAX_PERCENT)
        # Set availability status
        PROM_RESOURCE_OK.labels(resource_type='cpu').set(1 if cpu_ok else 0)
        PROM_RESOURCE_OK.labels(resource_type='memory').set(1 if mem_ok else 0)
        PROM_RESOURCE_OK.labels(resource_type='disk').set(1 if disk_ok else 0)
        PROM_RESOURCE_OK.labels(resource_type='gpu').set(1 if gpu_ok else 0)
        # Set _used_ metrics for alerting rules
        PROM_DISK_USED.set(disk_pct)
        PROM_MEMORY_USED.set(mem_pct)
        PROM_CPU_USED.set(cpu_pct)
        PROM_GPU_USED.set(gpu_pct)
        # Set degradation level (need to calculate here to avoid circular import)
        max_ratio = max(
            disk_pct / LIMITS.DISK_MAX_PERCENT,
            mem_pct / LIMITS.MEMORY_MAX_PERCENT,
            cpu_pct / LIMITS.CPU_MAX_PERCENT if cpu_pct > 0 else 0,
        )
        if max_ratio < 0.7:
            degradation = 0
        elif max_ratio < 0.85:
            degradation = 1
        elif max_ratio < 0.95:
            degradation = 2
        elif max_ratio < 1.0:
            degradation = 3
        else:
            degradation = 4
        PROM_DEGRADATION_LEVEL.set(degradation)

    return {
        "disk": {
            "used_percent": round(disk_pct, 1),
            "available_gb": round(disk_avail, 1),
            "total_gb": round(disk_total, 1),
            "ok": disk_ok,
            "limit_percent": LIMITS.DISK_MAX_PERCENT,
        },
        "memory": {
            "used_percent": round(mem_pct, 1),
            "available_gb": round(mem_avail, 1),
            "total_gb": round(mem_total, 1),
            "ok": mem_ok,
            "limit_percent": LIMITS.MEMORY_MAX_PERCENT,
        },
        "cpu": {
            "used_percent": round(cpu_pct, 1),
            "load_per_cpu": round(load_per_cpu, 2),
            "cpu_count": cpu_count,
            "ok": cpu_ok,
            "limit_percent": LIMITS.CPU_MAX_PERCENT,
        },
        "gpu": {
            "used_percent": round(gpu_pct, 1),
            "available_gb": round(gpu_avail, 1),
            "total_gb": round(gpu_total, 1),
            "ok": gpu_ok,
            "limit_percent": LIMITS.GPU_MAX_PERCENT,
        },
        "can_proceed": can_proceed(check_gpu=False),
    }


def print_resource_status() -> None:
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


# ============================================
# Async Resource Limiter
# ============================================

class AsyncResourceLimiter:
    """Async resource limiter with automatic exponential backoff.

    Provides both sync and async interfaces for waiting on resources.

    Usage:
        limiter = AsyncResourceLimiter()

        # Async
        await limiter.wait_for_resources_async("training")

        # Sync
        limiter.wait_for_resources_sync("selfplay")

        # Context manager
        async with limiter.acquire("my_task"):
            await do_work()
    """

    MIN_BACKOFF_SECONDS = 5.0
    MAX_BACKOFF_SECONDS = 300.0
    BACKOFF_MULTIPLIER = 1.5

    def __init__(
        self,
        disk_required_gb: float = 2.0,
        mem_required_gb: float = 1.0,
        gpu_required_gb: float = 0.0,
        max_concurrent_tasks: int = 0,
    ):
        self.disk_required_gb = disk_required_gb
        self.mem_required_gb = mem_required_gb
        self.gpu_required_gb = gpu_required_gb
        self.max_concurrent_tasks = max_concurrent_tasks
        self._active_tasks: dict = {}
        self._backoff_seconds = self.MIN_BACKOFF_SECONDS

    def _check_resources(self) -> tuple:
        """Check if resources are available.

        Returns:
            Tuple of (ok, list_of_issues)
        """
        issues = []

        if not check_disk_space(self.disk_required_gb, log_warning=False):
            disk_pct, _, _ = get_disk_usage()
            issues.append(f"Disk: {disk_pct:.1f}%")

        if not check_memory(self.mem_required_gb, log_warning=False):
            mem_pct, _, _ = get_memory_usage()
            issues.append(f"Memory: {mem_pct:.1f}%")

        if not check_cpu(log_warning=False):
            cpu_pct, _, _ = get_cpu_usage()
            issues.append(f"CPU: {cpu_pct:.1f}%")

        if self.gpu_required_gb > 0 and not check_gpu_memory(self.gpu_required_gb, log_warning=False):
            gpu_pct, _, _ = get_gpu_memory_usage()
            issues.append(f"GPU: {gpu_pct:.1f}%")

        if self.max_concurrent_tasks > 0 and len(self._active_tasks) >= self.max_concurrent_tasks:
            issues.append(f"Tasks: {len(self._active_tasks)}/{self.max_concurrent_tasks}")

        return len(issues) == 0, issues

    async def wait_for_resources_async(
        self,
        task_name: str = "unknown",
        max_wait_seconds: float = 600.0,
    ) -> bool:
        """Async wait until resources are available.

        Args:
            task_name: Name for logging
            max_wait_seconds: Maximum wait time

        Returns:
            True if resources available, False if timeout
        """
        import asyncio

        total_waited = 0.0
        backoff = self.MIN_BACKOFF_SECONDS

        while total_waited < max_wait_seconds:
            ok, issues = self._check_resources()

            if ok:
                self._backoff_seconds = self.MIN_BACKOFF_SECONDS
                return True

            logger.warning(
                f"[ResourceLimiter] Task '{task_name}' backing off for {backoff:.0f}s. "
                f"Issues: {', '.join(issues)}"
            )

            await asyncio.sleep(backoff)
            total_waited += backoff
            backoff = min(backoff * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF_SECONDS)

        logger.error(f"[ResourceLimiter] Task '{task_name}' timed out after {max_wait_seconds}s")
        return False

    def wait_for_resources_sync(
        self,
        task_name: str = "unknown",
        max_wait_seconds: float = 600.0,
    ) -> bool:
        """Sync wait until resources are available.

        Args:
            task_name: Name for logging
            max_wait_seconds: Maximum wait time

        Returns:
            True if resources available, False if timeout
        """
        total_waited = 0.0
        backoff = self.MIN_BACKOFF_SECONDS

        while total_waited < max_wait_seconds:
            ok, issues = self._check_resources()

            if ok:
                self._backoff_seconds = self.MIN_BACKOFF_SECONDS
                return True

            logger.warning(
                f"[ResourceLimiter] Task '{task_name}' backing off for {backoff:.0f}s. "
                f"Issues: {', '.join(issues)}"
            )

            time.sleep(backoff)
            total_waited += backoff
            backoff = min(backoff * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF_SECONDS)

        logger.error(f"[ResourceLimiter] Task '{task_name}' timed out after {max_wait_seconds}s")
        return False

    def acquire(self, task_name: str):
        """Async context manager to acquire resources.

        Usage:
            async with limiter.acquire("my_task"):
                await do_work()
        """
        return _AsyncResourceContext(self, task_name)

    def _register_task(self, task_name: str) -> None:
        self._active_tasks[task_name] = time.time()

    def _unregister_task(self, task_name: str) -> None:
        self._active_tasks.pop(task_name, None)


class _AsyncResourceContext:
    """Async context manager for resource acquisition."""

    def __init__(self, limiter: AsyncResourceLimiter, task_name: str):
        self.limiter = limiter
        self.task_name = task_name

    async def __aenter__(self):
        await self.limiter.wait_for_resources_async(self.task_name)
        self.limiter._register_task(self.task_name)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.limiter._unregister_task(self.task_name)
        if self.limiter.gpu_required_gb > 0:
            clear_gpu_memory()
        return False


# ============================================
# Decorator for Resource-Aware Functions
# ============================================

def respect_resource_limits(
    task_name: str | None = None,
    disk_gb: float = 2.0,
    mem_gb: float = 1.0,
    gpu_gb: float = 0.0,
    max_wait_seconds: float = 600.0,
):
    """Decorator that waits for resources before executing function.

    Usage:
        @respect_resource_limits(task_name="training", gpu_gb=4.0)
        async def train_model():
            ...

        @respect_resource_limits(disk_gb=5.0)
        def save_data():
            ...
    """
    import asyncio
    import functools

    def decorator(func):
        limiter = AsyncResourceLimiter(
            disk_required_gb=disk_gb,
            mem_required_gb=mem_gb,
            gpu_required_gb=gpu_gb,
        )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = task_name or func.__name__
            if limiter.wait_for_resources_sync(name, max_wait_seconds):
                return func(*args, **kwargs)
            else:
                raise RuntimeError(f"Resources not available for {name} after {max_wait_seconds}s")

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = task_name or func.__name__
            if await limiter.wait_for_resources_async(name, max_wait_seconds):
                return await func(*args, **kwargs)
            else:
                raise RuntimeError(f"Resources not available for {name} after {max_wait_seconds}s")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ============================================
# Graceful Degradation under Resource Constraints
# ============================================

class OperationPriority:
    """Priority levels for operations during resource constraints.

    When resources are limited, lower priority operations are paused first.
    Higher number = higher priority = more likely to continue running.
    """
    BACKGROUND = 0     # Optional cleanup, stats collection
    LOW = 1            # Extra selfplay, backfill tasks
    NORMAL = 2         # Regular selfplay, data generation
    HIGH = 3           # Training, model evaluation
    CRITICAL = 4       # Data sync, model promotion, health checks


def get_degradation_level() -> int:
    """Get current degradation level based on resource pressure.

    Returns:
        0: Normal - all operations can proceed
        1: Light pressure - pause BACKGROUND operations
        2: Moderate pressure - pause LOW priority operations
        3: Heavy pressure - pause NORMAL operations
        4: Critical - only CRITICAL operations allowed
    """
    disk_pct, _, _ = get_disk_usage()
    mem_pct, _, _ = get_memory_usage()
    cpu_pct, _, _ = get_cpu_usage()

    # Worst-case resource
    max_pct = max(
        disk_pct / LIMITS.DISK_MAX_PERCENT,  # Normalized to limit
        mem_pct / LIMITS.MEMORY_MAX_PERCENT,
        cpu_pct / LIMITS.CPU_MAX_PERCENT,
    )

    if max_pct < 0.7:  # Below 70% of limit
        return 0  # Normal
    elif max_pct < 0.85:  # 70-85% of limit
        return 1  # Light pressure
    elif max_pct < 0.95:  # 85-95% of limit
        return 2  # Moderate pressure
    elif max_pct < 1.0:  # 95-100% of limit
        return 3  # Heavy pressure
    else:  # Over limit
        return 4  # Critical


def should_proceed_with_priority(priority: int) -> bool:
    """Check if an operation with given priority should proceed.

    Args:
        priority: Operation priority (use OperationPriority constants)

    Returns:
        True if the operation should proceed, False if it should be paused

    Example:
        if should_proceed_with_priority(OperationPriority.LOW):
            run_backfill_selfplay()
        else:
            logger.info("Pausing backfill due to resource pressure")
    """
    degradation = get_degradation_level()

    # At degradation level N, operations with priority <= N are paused
    # CRITICAL (4) always proceeds unless at degradation level 5 (impossible)
    return priority > degradation


def get_recommended_actions() -> list:
    """Get list of recommended actions based on current resource state.

    Returns:
        List of string recommendations for the operator/system
    """
    status = get_resource_status()
    actions = []

    # Disk recommendations
    if not status["disk"]["ok"]:
        actions.append("CRITICAL: Disk above 70% - run disk cleanup immediately")
        actions.append("  - Remove old model checkpoints")
        actions.append("  - Clean up old JSONL files")
        actions.append("  - Remove database backups")
    elif status["disk"]["used_percent"] > 60:
        actions.append("WARNING: Disk at {:.0f}% - consider cleanup soon".format(
            status["disk"]["used_percent"]))

    # Memory recommendations
    if not status["memory"]["ok"]:
        actions.append("CRITICAL: Memory above 80% - reduce concurrent processes")
        actions.append("  - Stop low-priority selfplay")
        actions.append("  - Reduce batch sizes")
    elif status["memory"]["used_percent"] > 70:
        actions.append("WARNING: Memory at {:.0f}% - monitor closely".format(
            status["memory"]["used_percent"]))

    # CPU recommendations
    if not status["cpu"]["ok"]:
        actions.append("CRITICAL: CPU above 80% - reduce concurrent processes")
        actions.append("  - Pause background tasks")
        actions.append("  - Reduce selfplay parallelism")

    # GPU recommendations
    if not status["gpu"]["ok"]:
        actions.append("CRITICAL: GPU memory above 80% - reduce GPU workload")
        actions.append("  - Reduce batch sizes")
        actions.append("  - Stop extra GPU selfplay")

    if not actions:
        degradation = get_degradation_level()
        if degradation == 0:
            actions.append("OK: All resources within normal limits")
        else:
            actions.append(f"INFO: Degradation level {degradation} - some operations may be paused")

    return actions


if __name__ == "__main__":
    # Print status when run directly
    print_resource_status()

    # Also show degradation level and recommendations
    print("\n" + "=" * 50)
    print("GRACEFUL DEGRADATION STATUS")
    print("=" * 50)
    level = get_degradation_level()
    print(f"Degradation level: {level}")
    print(f"BACKGROUND operations: {'PAUSED' if level >= 1 else 'OK'}")
    print(f"LOW priority operations: {'PAUSED' if level >= 2 else 'OK'}")
    print(f"NORMAL operations: {'PAUSED' if level >= 3 else 'OK'}")
    print(f"HIGH priority operations: {'PAUSED' if level >= 4 else 'OK'}")
    print("CRITICAL operations: OK")

    print("\nRecommended actions:")
    for action in get_recommended_actions():
        print(f"  {action}")
