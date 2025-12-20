"""GPU Memory Profiling Utilities.

Provides tools for tracking tensor allocations and memory usage during
GPU operations. Useful for identifying memory leaks and optimizing
batch sizes.

Usage:
    from app.ai.gpu_memory_profiler import MemoryProfiler, profile_memory

    # Context manager for profiling a block
    with profile_memory("training_step") as prof:
        # ... do GPU work ...
    print(prof.report())

    # Or use the profiler directly
    profiler = MemoryProfiler()
    profiler.snapshot("before")
    # ... do work ...
    profiler.snapshot("after")
    print(profiler.compare("before", "after"))
"""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a point in time."""

    name: str
    timestamp: float
    allocated_bytes: int
    reserved_bytes: int
    max_allocated_bytes: int
    num_tensors: int
    tensor_sizes: dict[str, int] = field(default_factory=dict)

    @property
    def allocated_mb(self) -> float:
        return self.allocated_bytes / (1024 * 1024)

    @property
    def reserved_mb(self) -> float:
        return self.reserved_bytes / (1024 * 1024)

    @property
    def max_allocated_mb(self) -> float:
        return self.max_allocated_bytes / (1024 * 1024)


class MemoryProfiler:
    """Profiler for tracking GPU/MPS memory usage."""

    def __init__(self, device: torch.device | None = None):
        """Initialize the profiler.

        Args:
            device: Device to profile. If None, auto-detects.
        """
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        self.device = device
        self.snapshots: dict[str, MemorySnapshot] = {}
        self._is_cuda = device.type == "cuda"
        self._is_mps = device.type == "mps"

    def _get_memory_stats(self) -> tuple[int, int, int]:
        """Get current memory statistics.

        Returns:
            Tuple of (allocated_bytes, reserved_bytes, max_allocated_bytes)
        """
        if self._is_cuda:
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            max_allocated = torch.cuda.max_memory_allocated(self.device)
            return allocated, reserved, max_allocated
        elif self._is_mps:
            # MPS has limited memory introspection
            # Use driver API if available
            try:
                allocated = torch.mps.current_allocated_memory()
                # MPS doesn't have reserved/max tracking
                return allocated, allocated, allocated
            except AttributeError:
                return 0, 0, 0
        else:
            # CPU - no GPU memory to track
            return 0, 0, 0

    def _count_tensors(self) -> tuple[int, dict[str, int]]:
        """Count tensors and their sizes by dtype.

        Returns:
            Tuple of (total_count, sizes_by_dtype)
        """
        gc.collect()
        if self._is_cuda:
            torch.cuda.synchronize()

        count = 0
        sizes: dict[str, int] = {}

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.device == self.device:
                    count += 1
                    dtype_name = str(obj.dtype)
                    size = obj.numel() * obj.element_size()
                    sizes[dtype_name] = sizes.get(dtype_name, 0) + size
            except (ReferenceError, RuntimeError):
                pass

        return count, sizes

    def snapshot(self, name: str) -> MemorySnapshot:
        """Take a memory snapshot.

        Args:
            name: Name for this snapshot

        Returns:
            The captured snapshot
        """
        allocated, reserved, max_allocated = self._get_memory_stats()
        num_tensors, tensor_sizes = self._count_tensors()

        snap = MemorySnapshot(
            name=name,
            timestamp=time.time(),
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            max_allocated_bytes=max_allocated,
            num_tensors=num_tensors,
            tensor_sizes=tensor_sizes,
        )
        self.snapshots[name] = snap
        return snap

    def compare(self, before: str, after: str) -> dict[str, Any]:
        """Compare two snapshots.

        Args:
            before: Name of the before snapshot
            after: Name of the after snapshot

        Returns:
            Dictionary with comparison results
        """
        snap_before = self.snapshots.get(before)
        snap_after = self.snapshots.get(after)

        if snap_before is None or snap_after is None:
            return {"error": f"Snapshot not found: {before if snap_before is None else after}"}

        delta_allocated = snap_after.allocated_bytes - snap_before.allocated_bytes
        delta_reserved = snap_after.reserved_bytes - snap_before.reserved_bytes
        delta_tensors = snap_after.num_tensors - snap_before.num_tensors
        elapsed = snap_after.timestamp - snap_before.timestamp

        return {
            "before": before,
            "after": after,
            "elapsed_seconds": elapsed,
            "delta_allocated_mb": delta_allocated / (1024 * 1024),
            "delta_reserved_mb": delta_reserved / (1024 * 1024),
            "delta_tensors": delta_tensors,
            "before_allocated_mb": snap_before.allocated_mb,
            "after_allocated_mb": snap_after.allocated_mb,
            "peak_allocated_mb": snap_after.max_allocated_mb,
        }

    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        if self._is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def clear_cache(self) -> None:
        """Clear GPU cache to free unused memory."""
        gc.collect()
        if self._is_cuda:
            torch.cuda.empty_cache()
        elif self._is_mps:
            with suppress(AttributeError):
                torch.mps.empty_cache()

    def report(self) -> str:
        """Generate a human-readable report of all snapshots.

        Returns:
            Formatted report string
        """
        lines = ["=" * 60, "GPU Memory Profile Report", "=" * 60]

        if not self.snapshots:
            lines.append("No snapshots captured.")
            return "\n".join(lines)

        for name, snap in self.snapshots.items():
            lines.extend([
                f"\n{name}:",
                f"  Allocated: {snap.allocated_mb:.2f} MB",
                f"  Reserved:  {snap.reserved_mb:.2f} MB",
                f"  Peak:      {snap.max_allocated_mb:.2f} MB",
                f"  Tensors:   {snap.num_tensors}",
            ])

            if snap.tensor_sizes:
                lines.append("  By dtype:")
                for dtype, size in sorted(snap.tensor_sizes.items()):
                    lines.append(f"    {dtype}: {size / (1024 * 1024):.2f} MB")

        return "\n".join(lines)


@contextmanager
def profile_memory(name: str = "block", device: torch.device | None = None):
    """Context manager for profiling a code block's memory usage.

    Args:
        name: Name for this profiling block
        device: Device to profile

    Yields:
        MemoryProfiler instance with before/after snapshots

    Example:
        with profile_memory("training") as prof:
            model.train_step(batch)
        print(prof.compare("before", "after"))
    """
    profiler = MemoryProfiler(device)
    profiler.reset_peak_stats()
    profiler.snapshot("before")

    try:
        yield profiler
    finally:
        profiler.snapshot("after")


def get_tensor_memory_usage(device: torch.device | None = None) -> dict[str, Any]:
    """Get current tensor memory usage breakdown.

    Args:
        device: Device to analyze. If None, checks all devices.

    Returns:
        Dictionary with memory usage by tensor type
    """
    gc.collect()

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    result: dict[str, Any] = {
        "device": str(device),
        "tensors_by_dtype": {},
        "tensors_by_size": [],
        "total_count": 0,
        "total_bytes": 0,
    }

    tensors_info = []

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == device.type:
                dtype = str(obj.dtype)
                size_bytes = obj.numel() * obj.element_size()
                shape = tuple(obj.shape)

                result["total_count"] += 1
                result["total_bytes"] += size_bytes

                if dtype not in result["tensors_by_dtype"]:
                    result["tensors_by_dtype"][dtype] = {"count": 0, "bytes": 0}
                result["tensors_by_dtype"][dtype]["count"] += 1
                result["tensors_by_dtype"][dtype]["bytes"] += size_bytes

                tensors_info.append({
                    "dtype": dtype,
                    "shape": shape,
                    "bytes": size_bytes,
                })
        except (ReferenceError, RuntimeError):
            pass

    # Sort by size and take top 10
    tensors_info.sort(key=lambda x: x["bytes"], reverse=True)
    result["tensors_by_size"] = tensors_info[:10]
    result["total_mb"] = result["total_bytes"] / (1024 * 1024)

    return result
