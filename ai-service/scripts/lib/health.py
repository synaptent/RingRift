"""Health check utilities for scripts.

Provides common patterns for system and service health monitoring:
- System resource checks (CPU, memory, disk)
- Service health checks (HTTP endpoints)
- Process health verification
- GPU monitoring

Usage:
    from scripts.lib.health import (
        SystemHealth,
        check_system_health,
        check_disk_space,
        check_memory,
        check_http_health,
        check_process_health,
    )

    # Check overall system health
    health = check_system_health()
    if health.disk_percent > 80:
        print("Disk space low!")

    # Check specific service
    if check_http_health("http://localhost:8770/health"):
        print("Service is healthy")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from scripts.lib.process import is_process_running

logger = logging.getLogger(__name__)


@dataclass
class DiskHealth:
    """Disk health information.

    Attributes:
        path: Path being checked
        total_bytes: Total disk space
        used_bytes: Used disk space
        free_bytes: Free disk space
        percent_used: Percentage used (0-100)
    """
    path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    percent_used: float

    @property
    def total_gb(self) -> float:
        """Total space in GB."""
        return self.total_bytes / (1024 ** 3)

    @property
    def used_gb(self) -> float:
        """Used space in GB."""
        return self.used_bytes / (1024 ** 3)

    @property
    def free_gb(self) -> float:
        """Free space in GB."""
        return self.free_bytes / (1024 ** 3)

    @property
    def is_critical(self) -> bool:
        """Check if disk usage is critical (>90% or <2GB free)."""
        return self.percent_used > 90 or self.free_gb < 2.0

    @property
    def is_warning(self) -> bool:
        """Check if disk usage is warning level (>70%)."""
        return self.percent_used > 70


@dataclass
class MemoryHealth:
    """Memory health information.

    Attributes:
        total_bytes: Total system memory
        available_bytes: Available memory
        used_bytes: Used memory
        percent_used: Percentage used (0-100)
    """
    total_bytes: int
    available_bytes: int
    used_bytes: int
    percent_used: float

    @property
    def total_gb(self) -> float:
        """Total memory in GB."""
        return self.total_bytes / (1024 ** 3)

    @property
    def available_gb(self) -> float:
        """Available memory in GB."""
        return self.available_bytes / (1024 ** 3)

    @property
    def is_critical(self) -> bool:
        """Check if memory usage is critical (>95%)."""
        return self.percent_used > 95

    @property
    def is_warning(self) -> bool:
        """Check if memory usage is warning level (>85%)."""
        return self.percent_used > 85


@dataclass
class CPUHealth:
    """CPU health information.

    Attributes:
        percent_used: Current CPU utilization (0-100)
        load_1min: 1-minute load average
        load_5min: 5-minute load average
        load_15min: 15-minute load average
        core_count: Number of CPU cores
    """
    percent_used: float
    load_1min: float
    load_5min: float
    load_15min: float
    core_count: int

    @property
    def load_per_core(self) -> float:
        """Load per CPU core (1-min average)."""
        return self.load_1min / self.core_count if self.core_count > 0 else 0

    @property
    def is_overloaded(self) -> bool:
        """Check if CPU is overloaded (load > 2x cores)."""
        return self.load_1min > (self.core_count * 2)


@dataclass
class GPUInfo:
    """Information about a single GPU.

    Attributes:
        index: GPU index
        name: GPU model name
        memory_total_mb: Total GPU memory
        memory_used_mb: Used GPU memory
        utilization_percent: GPU utilization (0-100)
        temperature_c: GPU temperature in Celsius
    """
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_percent: int
    temperature_c: int

    @property
    def memory_free_mb(self) -> int:
        """Free GPU memory in MB."""
        return self.memory_total_mb - self.memory_used_mb

    @property
    def memory_percent(self) -> float:
        """Memory utilization percentage."""
        if self.memory_total_mb > 0:
            return (self.memory_used_mb / self.memory_total_mb) * 100
        return 0.0


@dataclass
class SystemHealth:
    """Overall system health status.

    Attributes:
        timestamp: When health was checked
        disk: Disk health information
        memory: Memory health information
        cpu: CPU health information
        gpus: List of GPU information
        is_healthy: Overall health status
        warnings: List of warning messages
        errors: List of error messages
    """
    timestamp: float = field(default_factory=time.time)
    disk: DiskHealth | None = None
    memory: MemoryHealth | None = None
    cpu: CPUHealth | None = None
    gpus: list[GPUInfo] = field(default_factory=list)
    is_healthy: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def disk_percent(self) -> float:
        """Disk usage percentage."""
        return self.disk.percent_used if self.disk else 0.0

    @property
    def memory_percent(self) -> float:
        """Memory usage percentage."""
        return self.memory.percent_used if self.memory else 0.0

    @property
    def cpu_percent(self) -> float:
        """CPU usage percentage."""
        return self.cpu.percent_used if self.cpu else 0.0

    @property
    def gpu_count(self) -> int:
        """Number of GPUs."""
        return len(self.gpus)


@dataclass
class ServiceHealth:
    """Health status of a service.

    Attributes:
        name: Service name
        url: Service URL
        is_healthy: Whether service responded successfully
        response_time_ms: Response time in milliseconds
        status_code: HTTP status code
        error: Error message if unhealthy
        response_data: Parsed response data
    """
    name: str
    url: str
    is_healthy: bool
    response_time_ms: float = 0.0
    status_code: int = 0
    error: str = ""
    response_data: dict[str, Any] = field(default_factory=dict)


def check_disk_space(path: Union[str, Path] = "/") -> DiskHealth:
    """Check disk space for a path.

    Args:
        path: Path to check disk space for

    Returns:
        DiskHealth with disk information
    """
    path = str(path)
    try:
        usage = shutil.disk_usage(path)
        percent = (usage.used / usage.total) * 100 if usage.total > 0 else 0

        return DiskHealth(
            path=path,
            total_bytes=usage.total,
            used_bytes=usage.used,
            free_bytes=usage.free,
            percent_used=percent,
        )
    except OSError as e:
        logger.error(f"Failed to check disk space for {path}: {e}")
        return DiskHealth(
            path=path,
            total_bytes=0,
            used_bytes=0,
            free_bytes=0,
            percent_used=0,
        )


def check_memory() -> MemoryHealth:
    """Check system memory usage.

    Returns:
        MemoryHealth with memory information
    """
    try:
        # Try reading from /proc/meminfo (Linux)
        meminfo_path = Path("/proc/meminfo")
        if meminfo_path.exists():
            meminfo = {}
            with open(meminfo_path) as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        # Value is in kB
                        value = int(parts[1].strip().split()[0]) * 1024
                        meminfo[key] = value

            total = meminfo.get("MemTotal", 0)
            available = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            used = total - available
            percent = (used / total) * 100 if total > 0 else 0

            return MemoryHealth(
                total_bytes=total,
                available_bytes=available,
                used_bytes=used,
                percent_used=percent,
            )

        # Fallback: use vm_stat on macOS
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse vm_stat output
            page_size = 4096  # Default page size
            stats = {}
            for line in result.stdout.split("\n"):
                if ":" in line:
                    parts = line.split(":")
                    key = parts[0].strip()
                    value = parts[1].strip().rstrip(".")
                    try:
                        stats[key] = int(value)
                    except ValueError:
                        pass

            # Get physical memory size
            sysctl = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            total = int(sysctl.stdout.strip()) if sysctl.returncode == 0 else 0

            free_pages = stats.get("Pages free", 0)
            inactive_pages = stats.get("Pages inactive", 0)
            available = (free_pages + inactive_pages) * page_size
            used = total - available
            percent = (used / total) * 100 if total > 0 else 0

            return MemoryHealth(
                total_bytes=total,
                available_bytes=available,
                used_bytes=used,
                percent_used=percent,
            )

    except Exception as e:
        logger.error(f"Failed to check memory: {e}")

    return MemoryHealth(
        total_bytes=0,
        available_bytes=0,
        used_bytes=0,
        percent_used=0,
    )


def check_cpu() -> CPUHealth:
    """Check CPU usage and load.

    Returns:
        CPUHealth with CPU information
    """
    try:
        # Get load averages
        load_1, load_5, load_15 = os.getloadavg()

        # Get CPU count
        core_count = os.cpu_count() or 1

        # Estimate CPU percent from load (rough approximation)
        cpu_percent = min(100, (load_1 / core_count) * 100)

        return CPUHealth(
            percent_used=cpu_percent,
            load_1min=load_1,
            load_5min=load_5,
            load_15min=load_15,
            core_count=core_count,
        )

    except Exception as e:
        logger.error(f"Failed to check CPU: {e}")

    return CPUHealth(
        percent_used=0,
        load_1min=0,
        load_5min=0,
        load_15min=0,
        core_count=os.cpu_count() or 1,
    )


def check_gpus() -> list[GPUInfo]:
    """Check GPU status using nvidia-smi.

    Returns:
        List of GPUInfo for each GPU
    """
    gpus = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    gpus.append(GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        memory_total_mb=int(parts[2]),
                        memory_used_mb=int(parts[3]),
                        utilization_percent=int(parts[4]),
                        temperature_c=int(parts[5]),
                    ))

    except FileNotFoundError:
        # nvidia-smi not available
        pass
    except Exception as e:
        logger.debug(f"Failed to check GPUs: {e}")

    return gpus


def check_system_health(
    disk_path: Union[str, Path] = "/",
    include_gpus: bool = True,
) -> SystemHealth:
    """Check overall system health.

    Args:
        disk_path: Path to check disk space for
        include_gpus: Whether to check GPU status

    Returns:
        SystemHealth with all system information
    """
    health = SystemHealth()

    # Check disk
    health.disk = check_disk_space(disk_path)
    if health.disk.is_critical:
        health.errors.append(f"Disk critical: {health.disk.percent_used:.1f}% used")
        health.is_healthy = False
    elif health.disk.is_warning:
        health.warnings.append(f"Disk warning: {health.disk.percent_used:.1f}% used")

    # Check memory
    health.memory = check_memory()
    if health.memory.is_critical:
        health.errors.append(f"Memory critical: {health.memory.percent_used:.1f}% used")
        health.is_healthy = False
    elif health.memory.is_warning:
        health.warnings.append(f"Memory warning: {health.memory.percent_used:.1f}% used")

    # Check CPU
    health.cpu = check_cpu()
    if health.cpu.is_overloaded:
        health.warnings.append(f"CPU overloaded: load {health.cpu.load_1min:.1f}")

    # Check GPUs
    if include_gpus:
        health.gpus = check_gpus()

    return health


def check_http_health(
    url: str,
    timeout: float = 10.0,
    expected_status: int = 200,
    json_field: str | None = None,
    headers: dict[str, str] | None = None,
) -> ServiceHealth:
    """Check health of an HTTP service.

    Args:
        url: URL to check
        timeout: Request timeout in seconds
        expected_status: Expected HTTP status code
        json_field: JSON field to check for health (e.g., "healthy", "status")
        headers: Additional headers to send

    Returns:
        ServiceHealth with service status
    """
    name = url.split("/")[-1] or "service"
    start_time = time.time()

    try:
        req = urllib.request.Request(url)
        if headers:
            for key, value in headers.items():
                req.add_header(key, value)

        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_time = (time.time() - start_time) * 1000
            status_code = response.status

            data = {}
            try:
                content = response.read().decode()
                data = json.loads(content)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

            # Check if healthy
            is_healthy = status_code == expected_status
            if json_field and data:
                field_value = data.get(json_field)
                if isinstance(field_value, bool):
                    is_healthy = is_healthy and field_value
                elif isinstance(field_value, str):
                    is_healthy = is_healthy and field_value.lower() in ("ok", "healthy", "true")

            return ServiceHealth(
                name=name,
                url=url,
                is_healthy=is_healthy,
                response_time_ms=response_time,
                status_code=status_code,
                response_data=data,
            )

    except urllib.error.HTTPError as e:
        return ServiceHealth(
            name=name,
            url=url,
            is_healthy=False,
            response_time_ms=(time.time() - start_time) * 1000,
            status_code=e.code,
            error=str(e.reason),
        )
    except Exception as e:
        return ServiceHealth(
            name=name,
            url=url,
            is_healthy=False,
            response_time_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )


def check_port_open(
    host: str,
    port: int,
    timeout: float = 5.0,
) -> bool:
    """Check if a TCP port is open.

    Args:
        host: Hostname or IP
        port: Port number
        timeout: Connection timeout

    Returns:
        True if port is open
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def check_process_health(
    pid: int | None = None,
    pattern: str | None = None,
    min_count: int = 1,
) -> tuple[bool, int]:
    """Check if processes are running.

    Args:
        pid: Specific PID to check
        pattern: Process name pattern to search for
        min_count: Minimum number of matching processes

    Returns:
        Tuple of (is_healthy, count)
    """
    if pid is not None:
        running = is_process_running(pid)
        return running, 1 if running else 0

    if pattern:
        from scripts.lib.process import count_processes_by_pattern
        count = count_processes_by_pattern(pattern)
        return count >= min_count, count

    return False, 0


def wait_for_healthy(
    check_fn: callable,
    timeout: float = 30.0,
    interval: float = 1.0,
    *args,
    **kwargs,
) -> bool:
    """Wait for a health check to pass.

    Args:
        check_fn: Health check function that returns bool or object with is_healthy
        timeout: Maximum time to wait
        interval: Time between checks
        *args, **kwargs: Arguments to pass to check_fn

    Returns:
        True if check passed within timeout
    """
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            result = check_fn(*args, **kwargs)
            if isinstance(result, bool):
                if result:
                    return True
            elif hasattr(result, "is_healthy"):
                if result.is_healthy:
                    return True
            elif hasattr(result, "success") and result.success:
                return True
        except Exception as e:
            logger.debug(f"Health check failed: {e}")

        time.sleep(interval)

    return False
