"""Memory profiling and peak tracking for RingRift distributed operations.

This module provides:
- Peak memory tracking for processes
- Memory sampling at intervals
- Process memory monitoring via RSS
- Memory report generation

Used to benchmark memory usage of heavy AI operations:
- Minimax with NNUE
- MCTS with neural network
- Descent optimization
- Self-play soaks
- CMA-ES optimization
- NNUE training
"""

from __future__ import annotations

import gc
import logging
import os
import platform
import resource
import subprocess
import threading
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MemorySample:
    """A single memory sample."""
    timestamp: float  # time.time()
    rss_mb: int  # Resident Set Size in MB
    peak_rss_mb: int  # Peak RSS so far
    tracemalloc_current_mb: float | None = None  # Python allocations
    tracemalloc_peak_mb: float | None = None  # Peak Python allocations


@dataclass
class MemoryProfile:
    """Complete memory profile for an operation."""
    operation_name: str
    start_time: datetime
    end_time: datetime | None = None
    samples: list[MemorySample] = field(default_factory=list)
    peak_rss_mb: int = 0
    peak_tracemalloc_mb: float = 0.0
    baseline_rss_mb: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Get operation duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def memory_increase_mb(self) -> int:
        """Get memory increase from baseline."""
        return self.peak_rss_mb - self.baseline_rss_mb

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "baseline_rss_mb": self.baseline_rss_mb,
            "peak_rss_mb": self.peak_rss_mb,
            "memory_increase_mb": self.memory_increase_mb,
            "peak_tracemalloc_mb": self.peak_tracemalloc_mb,
            "num_samples": len(self.samples),
            "metadata": self.metadata,
        }


def get_current_rss_mb() -> int:
    """Get current process RSS (Resident Set Size) in MB.

    This is the actual physical memory used by the process.
    """
    try:
        # Use resource module for cross-platform support
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if platform.system() == "Darwin":
            # macOS returns bytes
            return int(usage.ru_maxrss / (1024 * 1024))
        else:
            # Linux returns kilobytes
            return int(usage.ru_maxrss / 1024)
    except (OSError, AttributeError, ValueError):
        # OSError: resource unavailable, AttributeError: missing constant,
        # ValueError: invalid conversion
        pass

    try:
        # Fallback: read /proc/self/status on Linux
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb // 1024
    except (OSError, FileNotFoundError, PermissionError, ValueError, IndexError):
        # FileNotFoundError/PermissionError: file access issues
        # IOError: read errors, ValueError/IndexError: parsing errors
        pass

    try:
        # Fallback: use ps command
        pid = os.getpid()
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) // 1024
    except (subprocess.SubprocessError, OSError, FileNotFoundError, ValueError):
        # SubprocessError: process execution issues, OSError/FileNotFoundError: ps not found
        # ValueError: invalid conversion
        pass

    return 0


def get_peak_rss_mb() -> int:
    """Get peak RSS from process start.

    Uses resource.getrusage which tracks the maximum resident set size.
    """
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if platform.system() == "Darwin":
            return int(usage.ru_maxrss / (1024 * 1024))
        else:
            return int(usage.ru_maxrss / 1024)
    except (OSError, AttributeError, ValueError):
        # OSError: resource unavailable, AttributeError: missing constant,
        # ValueError: invalid conversion
        return get_current_rss_mb()


def get_process_rss_mb(pid: int) -> int | None:
    """Get RSS for a specific process by PID.

    Args:
        pid: Process ID

    Returns:
        RSS in MB, or None if process not found
    """
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip()) // 1024
    except (subprocess.SubprocessError, OSError, FileNotFoundError, ValueError):
        # SubprocessError: process execution issues (includes TimeoutExpired)
        # OSError/FileNotFoundError: ps not found, ValueError: invalid int conversion
        pass
    return None


class MemoryTracker:
    """Track memory usage over time with periodic sampling.

    Usage:
        tracker = MemoryTracker("my_operation")
        tracker.start()
        # ... do memory-intensive work ...
        tracker.stop()
        profile = tracker.get_profile()
        print(f"Peak memory: {profile.peak_rss_mb} MB")
    """

    def __init__(
        self,
        operation_name: str,
        sample_interval: float = 1.0,
        use_tracemalloc: bool = False,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize memory tracker.

        Args:
            operation_name: Name for this operation/benchmark
            sample_interval: Seconds between samples (default 1.0)
            use_tracemalloc: Also track Python allocations (adds overhead)
            metadata: Additional metadata to include in profile
        """
        self.operation_name = operation_name
        self.sample_interval = sample_interval
        self.use_tracemalloc = use_tracemalloc
        self.metadata = metadata or {}

        self._samples: list[MemorySample] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None
        self._baseline_rss: int = 0
        self._peak_rss: int = 0

    def start(self) -> None:
        """Start tracking memory usage."""
        if self._running:
            return

        # Force garbage collection for accurate baseline
        gc.collect()

        self._baseline_rss = get_current_rss_mb()
        self._peak_rss = self._baseline_rss
        self._start_time = datetime.now()
        self._samples = []
        self._running = True

        if self.use_tracemalloc:
            tracemalloc.start()

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

        logger.info(f"Started memory tracking for '{self.operation_name}' (baseline: {self._baseline_rss} MB)")

    def stop(self) -> MemoryProfile:
        """Stop tracking and return the memory profile."""
        self._running = False
        self._end_time = datetime.now()

        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        # Take final sample
        self._take_sample()

        tracemalloc_peak = 0.0
        if self.use_tracemalloc:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc_peak = peak / (1024 * 1024)
            tracemalloc.stop()

        profile = MemoryProfile(
            operation_name=self.operation_name,
            start_time=self._start_time or datetime.now(),
            end_time=self._end_time,
            samples=self._samples.copy(),
            peak_rss_mb=self._peak_rss,
            peak_tracemalloc_mb=tracemalloc_peak,
            baseline_rss_mb=self._baseline_rss,
            metadata=self.metadata,
        )

        logger.info(
            f"Stopped memory tracking for '{self.operation_name}': "
            f"peak={self._peak_rss} MB, increase={profile.memory_increase_mb} MB"
        )

        return profile

    def _sample_loop(self) -> None:
        """Background loop to take memory samples."""
        while self._running:
            self._take_sample()
            time.sleep(self.sample_interval)

    def _take_sample(self) -> None:
        """Take a single memory sample."""
        rss = get_current_rss_mb()
        peak_rss = max(self._peak_rss, rss)
        self._peak_rss = peak_rss

        tracemalloc_current = None
        tracemalloc_peak = None
        if self.use_tracemalloc:
            try:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc_current = current / (1024 * 1024)
                tracemalloc_peak = peak / (1024 * 1024)
            except RuntimeError:
                # tracemalloc raises RuntimeError if not started
                pass

        sample = MemorySample(
            timestamp=time.time(),
            rss_mb=rss,
            peak_rss_mb=peak_rss,
            tracemalloc_current_mb=tracemalloc_current,
            tracemalloc_peak_mb=tracemalloc_peak,
        )
        self._samples.append(sample)

    def get_current_rss(self) -> int:
        """Get current RSS memory in MB."""
        return get_current_rss_mb()

    def get_peak_rss(self) -> int:
        """Get peak RSS memory in MB since tracking started."""
        return self._peak_rss

    def __enter__(self) -> MemoryTracker:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def profile_function(
    func: Callable,
    operation_name: str | None = None,
    sample_interval: float = 0.5,
    use_tracemalloc: bool = False,
) -> tuple[Any, MemoryProfile]:
    """Profile memory usage of a function call.

    Args:
        func: Function to call (takes no arguments)
        operation_name: Name for the profile (defaults to function name)
        sample_interval: Seconds between memory samples
        use_tracemalloc: Also track Python allocations

    Returns:
        Tuple of (function result, MemoryProfile)

    Example:
        def heavy_computation():
            data = [list(range(10000)) for _ in range(1000)]
            return sum(sum(row) for row in data)

        result, profile = profile_function(heavy_computation)
        print(f"Result: {result}, Peak memory: {profile.peak_rss_mb} MB")
    """
    name = operation_name or getattr(func, "__name__", "unknown")

    tracker = MemoryTracker(
        operation_name=name,
        sample_interval=sample_interval,
        use_tracemalloc=use_tracemalloc,
    )

    tracker.start()
    try:
        result = func()
    finally:
        profile = tracker.stop()

    return result, profile


class RemoteMemoryMonitor:
    """Monitor memory usage on a remote host via SSH.

    Usage:
        from app.distributed.hosts import get_ssh_executor

        executor = get_ssh_executor("mac-studio")
        monitor = RemoteMemoryMonitor(executor, "selfplay")
        monitor.start()
        # ... remote process running ...
        samples = monitor.get_samples()
        monitor.stop()
    """

    def __init__(
        self,
        ssh_executor,  # SSHExecutor from hosts.py
        process_pattern: str,
        sample_interval: float = 5.0,
    ):
        """Initialize remote memory monitor.

        Args:
            ssh_executor: SSHExecutor instance for the remote host
            process_pattern: Pattern to match in ps output (e.g., "selfplay", "minimax")
            sample_interval: Seconds between samples
        """
        self.ssh_executor = ssh_executor
        self.process_pattern = process_pattern
        self.sample_interval = sample_interval

        self._samples: list[tuple[float, int]] = []  # (timestamp, rss_mb)
        self._running = False
        self._thread: threading.Thread | None = None
        self._peak_rss: int = 0

    def start(self) -> None:
        """Start monitoring remote process memory."""
        if self._running:
            return

        self._running = True
        self._samples = []
        self._peak_rss = 0

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

        logger.info(f"Started remote memory monitoring for pattern '{self.process_pattern}'")

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None

    def _sample_loop(self) -> None:
        """Background loop to sample remote memory."""
        while self._running:
            try:
                rss = self.ssh_executor.get_process_memory(self.process_pattern)
                if rss is not None:
                    self._samples.append((time.time(), rss))
                    self._peak_rss = max(self._peak_rss, rss)
            except Exception as e:
                logger.debug(f"Remote memory sample failed: {e}")

            time.sleep(self.sample_interval)

    def get_samples(self) -> list[tuple[float, int]]:
        """Get all memory samples as (timestamp, rss_mb) tuples."""
        return self._samples.copy()

    def get_peak_rss(self) -> int:
        """Get peak RSS observed."""
        return self._peak_rss

    def __enter__(self) -> RemoteMemoryMonitor:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def format_memory_profile(profile: MemoryProfile) -> str:
    """Format a memory profile as a readable string."""
    lines = [
        f"Memory Profile: {profile.operation_name}",
        f"  Duration: {profile.duration_seconds:.1f}s",
        f"  Baseline RSS: {profile.baseline_rss_mb} MB",
        f"  Peak RSS: {profile.peak_rss_mb} MB",
        f"  Memory Increase: {profile.memory_increase_mb} MB",
    ]

    if profile.peak_tracemalloc_mb > 0:
        lines.append(f"  Peak Python Allocations: {profile.peak_tracemalloc_mb:.1f} MB")

    lines.append(f"  Samples: {len(profile.samples)}")

    if profile.metadata:
        lines.append("  Metadata:")
        for key, value in profile.metadata.items():
            lines.append(f"    {key}: {value}")

    return "\n".join(lines)


def write_memory_report(
    profiles: list[MemoryProfile],
    output_path: str,
    format: str = "json",
) -> None:
    """Write memory profiles to a report file.

    Args:
        profiles: List of memory profiles
        output_path: Output file path
        format: "json" or "text"
    """
    import json

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        data = {
            "generated_at": datetime.now().isoformat(),
            "profiles": [p.to_dict() for p in profiles],
            "summary": {
                "total_profiles": len(profiles),
                "max_peak_rss_mb": max(p.peak_rss_mb for p in profiles) if profiles else 0,
                "max_memory_increase_mb": max(p.memory_increase_mb for p in profiles) if profiles else 0,
            },
        }
        with open(output, "w") as f:
            json.dump(data, f, indent=2)
    else:
        with open(output, "w") as f:
            f.write(f"Memory Report - Generated {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")

            for profile in profiles:
                f.write(format_memory_profile(profile))
                f.write("\n\n")

            f.write("Summary\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total profiles: {len(profiles)}\n")
            if profiles:
                f.write(f"Max peak RSS: {max(p.peak_rss_mb for p in profiles)} MB\n")
                f.write(f"Max memory increase: {max(p.memory_increase_mb for p in profiles)} MB\n")

    logger.info(f"Wrote memory report to {output_path}")
